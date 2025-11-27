import { auth } from "@/auth";
import { NextResponse } from "next/server";
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

function baseUrl() {
  if (process.env.NEXT_PUBLIC_APP_URL) return process.env.NEXT_PUBLIC_APP_URL!;
  if (process.env.VERCEL_URL) return `https://${process.env.VERCEL_URL}`;
  return "http://localhost:3000";
}

export async function POST(req: Request) {
  const session = await auth();
  // Only owners can provision new clients
  if (!session?.user || ((session.user as { role?: string }).role !== "OWNER")) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const { email, name } = await req.json();
  if (!email) return NextResponse.json({ error: "email required" }, { status: 400 });

  // Upsert the NextAuth user. Do NOT set role here — Prisma User has no 'role' column.
  const user = await prisma.user.upsert({
    where: { email },
    update: { name: name ?? undefined },
    create: { email, name: name ?? null, emailVerified: null },
  });

  // Send them a password-set email via your existing route
  try {
    await fetch(`${baseUrl()}/api/request-password-reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email }),
    });
  } catch {
    // ignore email errors in dev
  }

  return NextResponse.json({ ok: true, id: user.id });
}
