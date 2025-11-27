import { NextResponse } from "next/server"
import { PrismaClient } from "@prisma/client"
import { auth } from "@/auth"

const prisma = new PrismaClient()

export async function POST() {
  // Require an authenticated session
  const session = await auth()
  if (!session?.user?.email) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
  }

  try {
    // Clean up dependent rows first if your schema doesn't cascade
    await prisma.account.deleteMany({ where: { user: { email: session.user.email } } })
    await prisma.session.deleteMany({ where: { user: { email: session.user.email } } })
    await prisma.verificationToken.deleteMany({ where: { identifier: session.user.email } })
    await prisma.verificationToken.deleteMany({ where: { identifier: { startsWith: `reset:${session.user.email}` } } })

    // Finally delete the user
    await prisma.user.delete({ where: { email: session.user.email } })

    return NextResponse.json({ ok: true })
  } catch (e) {
    console.error("Delete account error:", e)
    return NextResponse.json({ error: "Server error" }, { status: 500 })
  }
}
