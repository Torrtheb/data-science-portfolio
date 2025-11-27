export const runtime = "nodejs"

import { NextResponse } from "next/server"
import { PrismaClient, Role } from "@prisma/client"
import { z } from "zod"
import { auth } from "@/auth"

const prisma = new PrismaClient()

const Body = z.object({
  email: z.string().email(),
  role: z.enum(["OWNER", "STAFF", "CLIENT"]),
})

export async function POST(req: Request) {
  const session = await auth()
  if (!session || session.user.role !== "OWNER") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 })
  }

  const json = await req.json().catch(() => null)
  const parsed = Body.safeParse(json)
  if (!parsed.success) {
    return NextResponse.json({ error: "Invalid input" }, { status: 400 })
  }

  const { email, role } = parsed.data

  await prisma.user.update({
    where: { email: email.toLowerCase() },
    data: { role: role as Role },
  })

  return NextResponse.json({ ok: true })
}
