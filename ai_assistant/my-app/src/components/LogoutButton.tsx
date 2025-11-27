"use client";

import { signOut } from "next-auth/react";
import { useSession } from "next-auth/react";

export default function LogoutButton() {
  const { status } = useSession();
  if (status !== "authenticated") return null;
  return (
    <button
      type="button"
      onClick={() => signOut({ callbackUrl: "/login" })}
      className="text-sm underline"
    >
      Sign out
    </button>
  );
}
