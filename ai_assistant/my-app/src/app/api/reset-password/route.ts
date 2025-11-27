// app/api/reset-password/route.ts
export const runtime = "nodejs";

import { NextResponse } from "next/server";
import { PrismaClient } from "@prisma/client";
import bcrypt from "bcryptjs";
import { sendPasswordChangedNotice } from "@/lib/mail";
import { passwordSchema } from "@/lib/password";

const prisma = new PrismaClient();

export async function POST(req: Request) {
  const { token, email, password } = await req.json().catch(() => ({}));

  if (!token || !email || !password) {
    return NextResponse.json({ error: "Invalid" }, { status: 400 });
  }

  const canonicalEmail = (email as string).trim().toLowerCase();

  const parsed = passwordSchema.safeParse(password);
  if (!parsed.success) {
    const msg = parsed.error.issues.map((i) => i.message).join(", ");
    return NextResponse.json({ error: msg }, { status: 400 });
  }

  const record = await prisma.verificationToken.findFirst({
    where: { token, identifier: `reset:${canonicalEmail}` },
  });
  if (!record || record.expires < new Date()) {
    return NextResponse.json({ error: "Bad token" }, { status: 400 });
  }

  const hash = await bcrypt.hash(password, 12);

  await prisma.$transaction([
    prisma.user.update({
      where: { email: canonicalEmail },
      data: {
        password: hash,             // you use 'password' column (not passwordHash)
        emailVerified: new Date(),  // mark verified so credentials can sign in
      },
    }),
    prisma.verificationToken.deleteMany({ where: { identifier: `reset:${canonicalEmail}` } }),
    prisma.session?.deleteMany
      ? prisma.session.deleteMany({ where: { user: { email: canonicalEmail } } })
      : prisma.verificationToken.findFirst({ where: { token: "__noop__" } }),
  ]);

  try {
    await sendPasswordChangedNotice(canonicalEmail);
  } catch {}

  return NextResponse.json({ ok: true });
}
