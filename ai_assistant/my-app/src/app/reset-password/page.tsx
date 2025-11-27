// app/reset-password/page.tsx
"use client"

import { useState, FormEvent } from "react"
import { useSearchParams } from "next/navigation"

function validatePassword(pw: string): string | null {
  if (pw.length < 10) return "At least 10 characters."
  if (pw.length > 64) return "Max 64 characters."
  if (!/[a-z]/.test(pw)) return "Include a lowercase letter."
  if (!/[A-Z]/.test(pw)) return "Include an uppercase letter."
  if (!/[0-9]/.test(pw)) return "Include a digit."
  if (!/[^A-Za-z0-9]/.test(pw)) return "Include a special character."
  if (pw.trim() !== pw) return "No leading or trailing spaces."
  // bcrypt 72-byte cap (bytes, not chars)
  if (new TextEncoder().encode(pw).length > 72) return "Too long for bcrypt."
  return null
}

export default function ResetPasswordPage() {
  const params = useSearchParams()
  const token = params.get("token") || ""
  const email = params.get("email") || ""
  const [pw, setPw] = useState("")
  const [pw2, setPw2] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [done, setDone] = useState(false)

  async function onSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)

    if (!token || !email) return setError("Invalid link.")
    if (pw !== pw2) return setError("Passwords do not match.")
    const policy = validatePassword(pw)
    if (policy) return setError(policy)

    setLoading(true)
    const res = await fetch("/api/reset-password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token, email, password: pw }),
    })
    setLoading(false)

    if (res.ok) {
      setDone(true)
    } else {
      const json = await res.json().catch(() => ({}))
      setError(json?.error || "Link invalid or expired.")
    }
  }

  if (!token || !email) return <main className="p-6">Invalid link.</main>

  if (done) {
    return (
      <main className="p-6">
        <div className="mx-auto max-w-sm rounded-xl border p-6">
          <h1 className="text-xl font-semibold">Password updated</h1>
          <p className="mt-2 text-sm text-neutral-600">
            You can now{" "}
            <a className="underline" href="/login">
              sign in
            </a>
            .
          </p>
        </div>
      </main>
    )
  }

  return (
    <main className="mx-auto max-w-sm p-6">
      <div className="rounded-2xl border p-6">
        <h1 className="text-2xl font-semibold mb-4">Reset password</h1>

        {error && (
          <div className="mb-3 border border-red-300 bg-red-50 text-red-700 p-3 rounded-md text-sm">
            {error}
          </div>
        )}

        <form onSubmit={onSubmit} className="space-y-3">
          <input
            className="w-full border rounded-md p-2"
            type="password"
            placeholder="New password"
            value={pw}
            onChange={(e) => setPw(e.target.value)}
          />
          <input
            className="w-full border rounded-md p-2"
            type="password"
            placeholder="Confirm new password"
            value={pw2}
            onChange={(e) => setPw2(e.target.value)}
          />
          <p className="text-xs text-neutral-500">
            Must be 10–64 chars, include upper, lower, digit, and special.
          </p>
          <button
            disabled={loading}
            className="w-full rounded-md bg-blue-600 px-3 py-2 text-white"
          >
            {loading ? "Updating..." : "Update password"}
          </button>
        </form>
      </div>
    </main>
  )
}
