"use client"
import { useState, FormEvent } from "react"

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("")
  const [sent, setSent] = useState(false)
  const [loading, setLoading] = useState(false)

  async function onSubmit(e: FormEvent) {
    e.preventDefault()
    setLoading(true)
    await fetch("/api/request-password-reset", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ email }),
    })
    setLoading(false)
    setSent(true) // avoid leaking which emails exist
  }

  return (
    <main className="mx-auto max-w-sm p-6">
      <h1 className="text-2xl font-semibold mb-4">Forgot password</h1>
      {sent ? (
        <p className="text-sm text-gray-600">If that email exists, we sent a reset link.</p>
      ) : (
        <form onSubmit={onSubmit} className="space-y-3">
          <input className="w-full border rounded-md p-2" type="email" placeholder="you@example.com" value={email} onChange={e=>setEmail(e.target.value)} required />
          <button disabled={loading} className="w-full border rounded-md p-2">{loading ? "Sending..." : "Send reset link"}</button>
        </form>
      )}
    </main>
  )
}
