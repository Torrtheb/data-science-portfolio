import { auth } from "@/auth";
import { apiFetch } from "@/lib/api";
import { cookies } from "next/headers";
import OwnerClientDetailClient from "./OwnerClientDetailClient";

export default async function ClientDetail({
  params,
}: {
  params: Promise<{ accountId: string }>;
}) {
  const session = await auth();
  if (!session?.user || ((session.user as { role?: string }).role !== "OWNER")) {
    return <div className="p-6">Forbidden.</div>;
  }

  const { accountId } = await params;

  const cookieStore = await cookies();
  const res = await apiFetch(`/api/back/owner/clients/${accountId}`, {
    cache: "no-store",
    headers: { cookie: cookieStore.toString() },
  });
  if (!res.ok) throw new Error("Failed to load client");
  const data = await res.json();

  const title =
    data.name ??
    data.client_name ??
    data.client_email ??
    `Account #${data.account_id}`;

  return (
    <main className="p-6 space-y-4 max-w-2xl">
      <h1 className="text-xl font-semibold">{title}</h1>
      <div className="text-sm opacity-80">Account ID: {data.account_id}</div>
      <OwnerClientDetailClient initial={data} />
    </main>
  );
}
