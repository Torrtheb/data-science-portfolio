import { auth } from "@/auth";
import { apiFetch } from "@/lib/api";
import { cookies } from "next/headers";
import OwnerClientsClient from "./OwnerClientsClient";

export default async function ClientsPage() {
  const session = await auth();
  if (!session?.user || ((session.user as { role?: string }).role !== "OWNER")) {
    return <div className="p-6">Forbidden.</div>;
  }

  try {
    const cookieStore = await cookies();
    // Proxy to backend scheduling router via Next API proxy
    const res = await apiFetch("/api/back/api/scheduling/owner/clients?include_people=false", {
      cache: "no-store",
      headers: { cookie: cookieStore.toString() },
    });
    if (!res.ok) {
      const detail = (await res.text()).trim();
      const target = res.headers.get("x-proxy-target");
      const status = res.status;
      const parts = [];
      if (detail) parts.push(detail);
      parts.push(`status ${status}`);
      if (target) parts.push(`target ${target}`);
      throw new Error(`Failed to load clients: ${parts.join(" | ")}`);
    }
    const initial = await res.json();

    return (
      <main className="p-6 space-y-4">
        <h1 className="text-2xl font-semibold">Clients</h1>
        <OwnerClientsClient initial={initial} />
      </main>
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Failed to load clients";
    return (
      <main className="p-6 space-y-4">
        <h1 className="text-2xl font-semibold">Clients</h1>
        <div className="rounded-md border border-red-200 bg-red-50 text-red-700 p-4">
          {msg}
        </div>
      </main>
    );
  }
}
