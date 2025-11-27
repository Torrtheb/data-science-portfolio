// src/app/api/account/timezone/route.ts
export const runtime = "nodejs";

import { NextResponse } from "next/server";
import { auth } from "@/auth";
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

export async function PATCH(req: Request) {
  const session = await auth();
  if (!session?.user?.email) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { timezone } = await req.json().catch(() => ({}));
  // Basic IANA sanity check: must contain a slash (e.g., America/Toronto)
  if (typeof timezone !== "string" || !timezone.includes("/")) {
    return NextResponse.json({ error: "Invalid timezone" }, { status: 400 });
  }

  await prisma.user.update({
    where: { email: session.user.email },
    data: { timezone }, // <- this needs the Prisma migration/generate done in step 0
  });

  // We just return ok. The UI will call router.refresh() to pull the new session.
  return NextResponse.json({ ok: true, timezone });
}
