// src/app/api/resend-verification/route.ts
import { NextResponse } from "next/server"
import { PrismaClient } from "@prisma/client"
import crypto from "crypto"
import { sendMail } from "@/lib/mail"

const prisma = new PrismaClient()

export async function POST(req: Request) {
  try {
    const { email } = await req.json().catch(() => ({}))
    if (!email || typeof email !== "string") {
      // Do not leak details
      return NextResponse.json({ ok: true })
    }

    const user = await prisma.user.findUnique({ where: { email } })
    if (!user || user.emailVerified) {
      // either no user or already verified → still pretend success
      return NextResponse.json({ ok: true })
    }

    // Clean up any prior verification tokens for this email
    await prisma.verificationToken.deleteMany({ where: { identifier: email } })

    // Issue fresh token (1h)
    const token = crypto.randomBytes(32).toString("hex")
    const expires = new Date(Date.now() + 60 * 60 * 1000)
    await prisma.verificationToken.create({
      data: { identifier: email, token, expires },
    })

    const baseUrl = process.env.AUTH_URL ?? "http://localhost:3000"
    const url = `${baseUrl}/verify-email?token=${encodeURIComponent(token)}&email=${encodeURIComponent(email)}`

    try {
      await sendMail({
        to: email,
        subject: "Verify your email",
        html: `<p>Click to verify your email:</p><p><a href="${url}">${url}</a></p><p>This link expires in 1 hour.</p>`,
      })
    } catch (err) {
      console.error("sendMail failed; manual link:", url, err)
    }

    console.log("Resent verification link:", url)
    return NextResponse.json({ ok: true })
  } catch (e) {
    console.error("resend-verification error:", e)
    // still don’t leak details
    return NextResponse.json({ ok: true })
  }
}