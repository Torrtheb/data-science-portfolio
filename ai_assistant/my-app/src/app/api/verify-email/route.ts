// app/api/verify-email/route.ts
import { NextResponse } from "next/server";
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

export async function POST(req: Request) {
  const { token, email } = await req.json();

  if (!token || !email) {
    return NextResponse.json({ error: "Missing token or email" }, { status: 400 });
  }

  const vt = await prisma.verificationToken.findFirst({
    where: { identifier: email, token },
  });

  if (!vt || vt.expires < new Date()) {
    return NextResponse.json({ error: "Invalid or expired token" }, { status: 400 });
  }

  // mark verified
  await prisma.user.update({
    where: { email },
    data: { emailVerified: new Date() },
  });

  // clean up all verification tokens for this email
  await prisma.verificationToken.deleteMany({ where: { identifier: email } });

  return NextResponse.json({ ok: true });
}
