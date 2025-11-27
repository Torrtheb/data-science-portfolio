// app/reset-password/reset-password-client.tsx
"use client";
import { useState } from "react";

export default function ResetPasswordClient({ token, email }: { token: string; email: string }) {
  const [pw, setPw] = useState("");
  const [confirm, setConfirm] = useState("");
  const [msg, setMsg] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setMsg(null);

    // basic client checks (server enforces again)
    if (pw !== confirm) return setMsg("Passwords do not match.");

    const res = await fetch("/api/reset-password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token, email, password: pw }),
    });
    const json = await res.json();
    if (!res.ok) return setMsg(json?.error ?? "Could not reset password.");
    setMsg("Password updated. You can sign in now.");
  }

  return (
    <form onSubmit={submit} className="mt-4 space-y-3">
      <input
        type="password"
        value={pw}
        onChange={(e) => setPw(e.target.value)}
        placeholder="New password"
        className="w-full rounded-md border px-3 py-2"
      />
      <input
        type="password"
        value={confirm}
        onChange={(e) => setConfirm(e.target.value)}
        placeholder="Confirm password"
        className="w-full rounded-md border px-3 py-2"
      />
      <button className="rounded-md bg-blue-600 px-3 py-2 text-white">Update password</button>
      {msg && <p className="text-sm text-neutral-700">{msg}</p>}
    </form>
  );
}
