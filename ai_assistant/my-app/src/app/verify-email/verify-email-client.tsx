// app/verify-email/verify-email-client.tsx
"use client";
import { useEffect, useState } from "react";
import Link from "next/link";

export default function VerifyEmailClient({ token, email }: { token: string; email: string }) {
  const [status, setStatus] = useState<"idle" | "ok" | "error">("idle");
  const [message, setMessage] = useState<string>("");

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const res = await fetch("/api/verify-email", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token, email }),
        });
        const json = await res.json();
        if (!alive) return;
        if (res.ok) {
          setStatus("ok");
          setMessage("Email verified! You can now sign in.");
        } else {
          setStatus("error");
          setMessage(json?.error ?? "Verification failed.");
        }
      } catch {
        if (!alive) return;
        setStatus("error");
        setMessage("Network error. Try again.");
      }
    })();
    return () => {
      alive = false;
    };
  }, [token, email]);

  if (status === "idle") {
    return <p className="text-sm">Working…</p>;
  }

  if (status === "ok") {
    return (
      <div className="rounded-lg bg-green-50 p-4 text-sm">
        <p className="font-medium text-green-700">{message}</p>
        <Link className="mt-3 inline-flex rounded-md border px-3 py-2 text-green-700" href="/login">
          Continue to sign in
        </Link>
      </div>
    );
  }

  return (
    <div className="rounded-lg bg-red-50 p-4 text-sm">
      <p className="font-medium text-red-700">{message}</p>
      <Link className="mt-3 inline-flex rounded-md border px-3 py-2" href="/resend-verification?email={encodeURIComponent(email)}">
        Resend verification
      </Link>
    </div>
  );
}
