// app/verify-email/page.tsx
import { PrismaClient } from "@prisma/client"
import { redirect } from "next/navigation"

const prisma = new PrismaClient()

export default async function VerifyEmailPage({
  // Next 15 requires awaiting searchParams; this is also safe on 14
  searchParams,
}: {
  searchParams:
    | Promise<{ token?: string; email?: string }>
    | { token?: string; email?: string }
}) {
  const sp = (await searchParams) as { token?: string; email?: string }
  const token = sp?.token
  const email = sp?.email

  if (!token || !email) {
    return (
      <main className="p-6">
        <div className="mx-auto max-w-md rounded-xl border p-6">
          <h1 className="text-lg font-semibold">Invalid verification link.</h1>
          <p className="mt-1 text-sm text-neutral-600">
            The link is missing required parameters.
          </p>
        </div>
      </main>
    )
  }

  // Look up the token + identifier together for safety
  const record = await prisma.verificationToken.findFirst({
    where: { token, identifier: email },
  })

  if (!record || record.expires < new Date()) {
    return (
      <main className="p-6">
        <div className="mx-auto max-w-md rounded-xl border p-6">
          <h1 className="text-lg font-semibold">Verification link invalid or expired.</h1>
          <p className="mt-1 text-sm text-neutral-600">
            Please request a new verification email.
          </p>
        </div>
      </main>
    )
  }

  // Atomically verify user and consume the token
  await prisma.$transaction([
    prisma.user.update({
      where: { email },
      data: { emailVerified: new Date() },
    }),
    prisma.verificationToken.deleteMany({ where: { identifier: email } }),
  ])

  redirect("/login?verified=1")
}
