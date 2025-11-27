// src/app/register/page.tsx
import { redirect } from "next/navigation";
export default function RegisterRedirect() {
  redirect("/login?error=InviteOnly");
}
