// src/lib/authz.ts
import { auth } from "@/auth"

type Role = "OWNER" | "STAFF" | "CLIENT";
type SessionWithRole = { user: { role: Role } };

export async function requireRole(required: Role) {
  const session = (await auth()) as SessionWithRole | null;
  if (!session || session.user.role !== required) {
    const err = new Error("FORBIDDEN");
    throw Object.assign(err, { status: 403 });
  }
  return session;
}

export async function requireAny(roles: Role[]) {
  const session = (await auth()) as SessionWithRole | null;
  if (!session || !roles.includes(session.user.role)) {
    const err = new Error("FORBIDDEN");
    throw Object.assign(err, { status: 403 });
  }
  return session;
}
