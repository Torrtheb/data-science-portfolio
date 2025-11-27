// my-app/src/app/api/delete-account/route.ts
import { NextResponse } from "next/server"
import { auth, signOut } from "@/auth"
import { PrismaClient } from "@prisma/client"

const prisma = new PrismaClient()

export async function POST() {
  const session = await auth()
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Not signed in" }, { status: 401 })
  }

  const userId = (session.user as { id?: string } | null | undefined)?.id as string
  try {
    await prisma.$transaction(async (tx) => {
      await tx.session.deleteMany({ where: { userId } })
      await tx.account.deleteMany({ where: { userId } })
      if (session.user?.email) {
        await tx.verificationToken.deleteMany({ where: { identifier: session.user.email } }).catch(() => {})
      }
      await tx.user.delete({ where: { id: userId } })
    })

    // Clears session cookies appropriately
    await signOut({ redirect: false })

    return NextResponse.json({ ok: true })
  } catch (e) {
    console.error(e)
    return NextResponse.json({ error: "Delete failed" }, { status: 500 })
  }
}
