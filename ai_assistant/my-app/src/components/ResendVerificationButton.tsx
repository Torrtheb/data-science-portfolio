"use client"

import { useState } from "react"

export default function ResendVerificationButton({ email }: { email: string }) {
  const [sending, setSending] = useState(false)
  const [sent, setSent] = useState(false)

  async function onClick() {
    if (!email) return alert("Enter your email first.")
    setSending(true)
    await fetch("/api/resend-verification", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email }),
    }).catch(() => {})
    setSending(false)
    setSent(true)
  }

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={sending || sent}
      className="border rounded-md px-3 py-2"
    >
      {sent ? "Sent!" : sending ? "Sending..." : "Resend verification email"}
    </button>
  )
}
