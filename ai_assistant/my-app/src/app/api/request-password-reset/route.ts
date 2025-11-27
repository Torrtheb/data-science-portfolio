// app/api/request-password-reset/route.ts
export const runtime = "nodejs"

import { NextResponse } from "next/server"
import { PrismaClient } from "@prisma/client"
import crypto from "crypto"
import { sendPasswordResetEmail } from "@/lib/mail"

const prisma = new PrismaClient()

export async function POST(req: Request) {
  const { email } = await req.json().catch(() => ({} as unknown as { email?: string }))
  // Enumeration-safe: always return 200
  if (!email || typeof email !== "string") return NextResponse.json({ ok: true })

  const canonicalEmail = email.trim().toLowerCase()

  const user = await prisma.user.findUnique({ where: { email: canonicalEmail } })
  if (!user) return NextResponse.json({ ok: true })

  // Throttle: if an unexpired reset token exists, don't spam emails
  const now = new Date()
  const existing = await prisma.verificationToken.findFirst({
    where: { identifier: `reset:${canonicalEmail}`, expires: { gt: now } },
    orderBy: { expires: "desc" },
  })
  if (existing) {
    return NextResponse.json({ ok: true })
  }

  const token = crypto.randomBytes(32).toString("hex")
  const expires = new Date(Date.now() + 24 * 60 * 60 * 1000) // 24 hours

  await prisma.verificationToken.create({
    data: { identifier: `reset:${canonicalEmail}`, token, expires },
  })

  const baseUrl =
    process.env.NEXT_PUBLIC_APP_URL ||
    process.env.AUTH_URL ||
    "http://localhost:3000"

  const targetEmail = user.email?.toLowerCase() || canonicalEmail

  const url = `${baseUrl}/reset-password?token=${encodeURIComponent(
    token
  )}&email=${encodeURIComponent(targetEmail)}`

  try {
    await sendPasswordResetEmail(targetEmail, url)
  } catch {
    // swallow to stay enumeration-safe
  }

  return NextResponse.json({ ok: true })
}
