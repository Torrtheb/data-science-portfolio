// src/app/dashboard/pricing/page.tsx
import { auth } from "@/auth";
import OwnerPricingClient from "./ui/OwnerPricingClient";

export const dynamic = "force-dynamic";

export default async function PricingPage() {
  const session = await auth();
  const role = (session?.user as { role?: string } | null | undefined)?.role;
  if (role !== "OWNER") return <div className="p-6">Forbidden.</div>;
  return (
    <div className="p-6">
      <OwnerPricingClient />
    </div>
  );
}
