import { auth } from "@/auth";
import { NextResponse } from "next/server";
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user || ((session.user as { role?: string }).role !== "OWNER")) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const { email } = await req.json();
  if (!email) return NextResponse.json({ error: "email required" }, { status: 400 });

  const user = await prisma.user.findUnique({ where: { email } });
  if (!user) return NextResponse.json({ ok: true });

  // Use the callback form of $transaction to avoid TS issues with optional promises
  await prisma.$transaction(async (tx) => {
    await tx.session.deleteMany({ where: { userId: user.id } });
    await tx.account.deleteMany({ where: { userId: user.id } });
    // Clear verification/reset tokens you use
    // Your reset flow uses identifier=`reset:${email}`
    await tx.verificationToken.deleteMany({
      where: { OR: [{ identifier: email }, { identifier: `reset:${email}` }] },
    }).catch(() => { /* model present in NextAuth adapter */ });

    await tx.user.delete({ where: { id: user.id } });
  });

  return NextResponse.json({ ok: true });
}
