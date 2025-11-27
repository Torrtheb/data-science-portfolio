// src/app/dashboard/page.tsx
import { auth } from "@/auth";
import { redirect } from "next/navigation";
import AssistantChat from "@/components/AssistantChat";
import ChatHistoryList from "@/components/ChatHistoryList";
import FunWelcome from "@/components/FunWelcome";
import Image from "next/image";

function CardLink({ href, title, desc }: { href: string; title: string; desc: string }) {
  return (
    <a href={href} className="block rounded-xl border p-4 hover:shadow-sm transition">
      <div className="font-medium mb-1">{title}</div>
      <div className="text-sm text-gray-600">{desc}</div>
    </a>
  );
}

export default async function DashboardPage() {
  const session = await auth();
  if (!session?.user) redirect("/login?error=CredentialsSignin");

  const role = (session.user as { role?: "OWNER" | "CLIENT" } | null | undefined)?.role as
    | "OWNER"
    | "CLIENT"
    | undefined;

  return (
    <main className="p-6 space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold">Hello, {session.user.name ?? "there"} 🎵</h1>
        <p className="text-gray-600">Signed in as {session.user.email}</p>
      </header>

      {role === "CLIENT" && (
        <div className="brand-banner rounded-xl p-4 flex items-center gap-4">
          <Image src="/piano-keys.svg" alt="Piano keys" width={48} height={48} />
          <div>
            <div className="font-medium" style={{ color: "rgb(var(--brand))" }}>Welcome to your studio dashboard</div>
            <div className="text-sm text-gray-600">Book lessons, see upcoming sessions, and manage your profile.</div>
          </div>
        </div>
      )}



      {role === "OWNER" ? (
        // OWNER: chat-first dashboard with history below (always stacked)
        <section className="space-y-6">
          <div>
            <AssistantChat />
          </div>
          <div className="space-y-3">
            <h2 className="text-lg font-medium">Previous chats</h2>
            {/* Show last 5 by default; user can toggle to show all inside the component */}
            <ChatHistoryList maxVisible={5} />
          </div>
        </section>
      ) : (
        // CLIENT: Appointments, Scheduling, Profile
        <section className="space-y-3">
          <h2 className="text-lg font-medium">Your account</h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <CardLink
              href="/client/appointments"
              title="Appointments"
              desc="See upcoming & past appointments."
            />
            <CardLink
              href="/scheduling"
              title="Scheduling"
              desc="Book a new appointment."
            />
            <CardLink
              href="/profile"
              title="Profile"
              desc="Manage your contact info and preferences."
            />
          </div>
        </section>
      )}
      {/* Fun, kid-friendly image widget for everyone */}
      <FunWelcome />
    </main>
  );
}
