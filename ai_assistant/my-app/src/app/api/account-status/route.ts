import { NextResponse } from "next/server"
import { PrismaClient } from "@prisma/client"

const prisma = new PrismaClient()

export async function POST(req: Request) {
  try {
    const { email } = await req.json()
    if (!email || typeof email !== "string") {
      return NextResponse.json({ error: "Invalid email" }, { status: 400 })
    }

    const user = await prisma.user.findUnique({
      where: { email },
      include: { accounts: true },
    })

    if (!user) {
      return NextResponse.json({
        exists: false,
        hasPassword: false,
        oauthProviders: [],
      })
    }

    const hasPassword = Boolean(user.password)
    const oauthProviders = user.accounts.map(a => a.provider) // e.g. ["google"]

    return NextResponse.json({
      exists: true,
      hasPassword,
      oauthProviders,
    })
  } catch {
    return NextResponse.json({ error: "Server error" }, { status: 500 })
  }
}
