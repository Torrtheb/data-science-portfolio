import { auth } from "@/auth";
import { redirect } from "next/navigation";

export default async function OwnerPage() {
  const session = await auth();
  if (!session) redirect("/login");
  if ((session.user as { role?: string } | null | undefined)?.role !== "OWNER") redirect("/dashboard");

  return (
    <main className="p-6">
      <h1 className="text-2xl font-semibold">Owner Console</h1>
      <p className="mt-2 opacity-80">Hello {session.user.name || "Owner"}.</p>

      <div className="mt-6 space-y-2">
        <a className="underline" href="/api/admin/secure">Test owner-only API</a>
      </div>
    </main>
  );
}
