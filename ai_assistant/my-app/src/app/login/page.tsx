"use client"

import { useState, useEffect, FormEvent } from "react"
import { signIn } from "next-auth/react"
import { useRouter, useSearchParams } from "next/navigation"
import Image from "next/image"
import type { ReactNode } from "react"
import ResendVerificationButton from "@/components/ResendVerificationButton"

type Status =
  | { kind: "idle" }
  | { kind: "error"; msg: string }
  | { kind: "hint"; msg: string; action?: ReactNode }
  | { kind: "success"; msg: string }

function mapAuthError(code: string | null): string | null {
  switch (code) {
    case "CredentialsSignin":
      return "Email or password is incorrect."
    case "OAuthAccountNotLinked":
      return "This email is linked to a different sign-in method."
    case "NoAccess":
      return "This Google account is not allowed."
    case "AccessDenied":
      return "Access denied."
    case "Configuration":
      return "Auth configuration error. Please contact support."
    case "Verification":
      return "Please verify your email first."
    default:
      return code ? "Sign-in failed." : null
  }
}

export default function LoginPage() {
  const router = useRouter()
  const params = useSearchParams()
  const urlError = params.get("error")
  const verified = params.get("verified")
  const reason = params.get("reason")
  const isVerificationError = urlError === "Verification" 

  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<Status>({ kind: "idle" })
  // Track the last auth error code so we can show field-level hints
  const [errCode, setErrCode] = useState<string | null>(null)
  const [hasInteracted, setHasInteracted] = useState(false)

  // Show a banner if redirected here with ?error=... or ?reason=...
  useEffect(() => {
    // Remember URL-driven errors for field-level rendering too
    if (urlError) setErrCode(urlError)

    if (urlError === "NoAccess") {
      // Show as error text only (no action button)
      const msg = mapAuthError(urlError)
      if (msg) setStatus({ kind: "error", msg })
      return
    }
    const msg = mapAuthError(urlError)
    if (msg) setStatus({ kind: "error", msg })
    else if (reason === "no-session" && !hasInteracted) {
      setStatus({ kind: "hint", msg: "Please sign in to continue." })
    }
  }, [urlError, reason, hasInteracted])

  // ✅ Show success when ?verified=1, then clear the query param after 4s
  useEffect(() => {
    if (verified === "1") {
      setStatus({ kind: "success", msg: "Email verified! You can log in now." })
      const t = setTimeout(() => {
        const url = new URL(window.location.href)
        url.searchParams.delete("verified")
        // keep other params (like ?error) intact
        router.replace(url.pathname + (url.search ? url.search : ""), { scroll: false })
      }, 4000)
      return () => clearTimeout(t)
    }
  }, [verified, router])

  // (Optional) Clear ?error and ?reason after 6s so they don't linger
  useEffect(() => {
    if (urlError || reason) {
      const t = setTimeout(() => {
        const url = new URL(window.location.href)
        url.searchParams.delete("error")
        url.searchParams.delete("reason")
        router.replace(url.pathname + (url.search ? url.search : ""), { scroll: false })
      }, 6000)
      return () => clearTimeout(t)
    }
  }, [urlError, reason, router])

  async function onSubmit(e: FormEvent) {
    e.preventDefault()
    setHasInteracted(true)
    setStatus({ kind: "idle" })
    setLoading(true)

    try {
      const res = await signIn("credentials", {
        email,
        password,
        redirect: false, // stay here
      })

      if (res?.ok) {
        router.push("/dashboard")
        return
      }

      // Show a friendly message immediately
      const code = res?.error ?? null
      setErrCode(code)
      const baseMsg = mapAuthError(code) ?? "Sign-in failed."
      setStatus({ kind: "error", msg: baseMsg })

      // Optional enrichment (no account? google-only?)
      try {
        const r = await fetch("/api/account-status", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email }),
        })
        if (!r.ok) return
        const data = await r.json()

        if (!data.exists) {
          setStatus({
            kind: "hint",
            msg: "No account found for this email.",
            action: <a className="underline" href="/register">Create an account</a>,
          })
        } else if (!data.hasPassword && Array.isArray(data.oauthProviders) && data.oauthProviders.includes("google")) {
          setStatus({
            kind: "hint",
            msg: "This email uses Google sign-in.",
            action: (
              <button
                type="button"
                onClick={() => signIn("google", { callbackUrl: "/dashboard", prompt: "select_account" })}
                className="border rounded-md px-3 py-2 mt-2"
              >
                Continue with Google
              </button>
            ),
          })
        }
      } catch {
        // keep base error
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-[80vh] flex items-center justify-center p-4 sm:p-6">
      <div className="w-full max-w-sm sm:max-w-md card p-5 sm:p-6 shadow-sm">
        <div className="mb-6 text-center">
          <div className="mx-auto mb-3 h-20 w-20 rounded-full flex items-center justify-center brand-banner">
            <Image src="/logo-studio.svg" alt="Studio logo" width={48} height={48} />
          </div>
          <h1 className="text-lg sm:text-xl font-semibold" style={{ color: "rgb(var(--brand))" }}>Welcome back</h1>
          <p className="text-sm text-gray-600">Sign in to your account</p>
        </div>

      {status.kind !== "idle" && (
        <div
            className={`mb-4 rounded-md border p-3 text-sm ${
            status.kind === "error"
                ? "border-red-300 bg-red-50 text-red-700"
                : status.kind === "success"
                ? "border-green-300 bg-green-50 text-green-700"
                : "border-blue-300 bg-blue-50 text-blue-700"
            }`}
        >
            <div className="flex items-center justify-between gap-3">
            <span>{status.msg}</span>

            {/* show resend button when verification is required */}
            {status.kind === "error" && isVerificationError && (
                <ResendVerificationButton email={email} />
            )}

            {status.kind === "hint" && status.action}
            </div>
        </div>
      )}


      {/* Live region for assistive tech */}
      <div aria-live="polite" className="sr-only">
        {status.kind === "error" ? status.msg : ""}
      </div>

      <form onSubmit={onSubmit} className="space-y-3" noValidate>
        <input
          className={`w-full border rounded-md px-3 py-3 ${errCode === "CredentialsSignin" ? "border-red-500" : ""}`}
          type="email"
          placeholder="you@example.com"
          value={email}
          onChange={(e) => { setEmail(e.target.value); if (errCode) setErrCode(null); if (status.kind !== "idle") setStatus({ kind: "idle" }); }}
          required
          autoComplete="email"
        />
        <input
          className={`w-full border rounded-md px-3 py-3 ${errCode === "CredentialsSignin" ? "border-red-500" : ""}`}
          type="password"
          placeholder="••••••••"
          value={password}
          onChange={(e) => { setPassword(e.target.value); if (errCode) setErrCode(null); if (status.kind !== "idle") setStatus({ kind: "idle" }); }}
          required
          autoComplete="current-password"
        />
        {errCode === "CredentialsSignin" && (
          <p className="text-sm text-red-600">Email or password is incorrect.</p>
        )}

        <button
          disabled={loading}
          className="w-full rounded-md px-3 py-3 font-medium text-white"
          style={{ backgroundColor: "rgb(var(--brand))" }}
          type="submit"
        >
          {loading ? "Signing in..." : "Sign in"}
        </button>
      </form>

      <div className="h-px bg-gray-200 my-6" />

      <button
        type="button"
        onClick={() => signIn("google", { callbackUrl: "/dashboard", prompt: "select_account" })}
        className="w-full border rounded-md px-3 py-3"
      >
        Continue with Google
      </button>

      <p className="mt-2 text-sm">
        <a href="/forgot-password" className="underline">Forgot password?</a>
      </p>
      </div>
    </main>
  )
}
