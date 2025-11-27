// middleware.ts
import { NextResponse, type NextRequest } from "next/server"

// OWNER-only route prefixes (UI hint only; server/API still enforce)
const OWNER_PREFIXES = ["/admin", "/dashboard/owner", "/dashboard/clients"]

// Public routes that should NOT require auth
const PUBLIC_PATHS = new Set([
  "/",
  "/login",
  "/register",
  "/verify-email",
  "/forgot-password",
  "/reset-password",
])

function hasSessionCookie(req: NextRequest): boolean {
  // NextAuth/Auth.js session cookie names in dev/prod
  // v4: next-auth.session-token / __Secure-next-auth.session-token
  // v5: authjs.session-token / __Secure-authjs.session-token
  return Boolean(
    req.cookies.get("__Secure-next-auth.session-token") ||
      req.cookies.get("next-auth.session-token") ||
      req.cookies.get("__Secure-authjs.session-token") ||
      req.cookies.get("authjs.session-token")
  )
}

export default function middleware(req: NextRequest) {
  const { pathname, origin, search } = req.nextUrl

  // Hard guards (extra safe)
  if (pathname.startsWith("/_next/")) return NextResponse.next()
  if (pathname.startsWith("/api/")) return NextResponse.next()
  if (pathname === "/favicon.ico") return NextResponse.next()

  // Redirect legacy register to login (invite-only)
  if (pathname === "/register") {
    const url = new URL("/login", origin)
    url.searchParams.set("error", "InviteOnly")
    return NextResponse.redirect(url)
  }

  // Public routes
  if (PUBLIC_PATHS.has(pathname)) return NextResponse.next()

  // Require presence of a NextAuth session cookie
  if (!hasSessionCookie(req)) {
    const url = new URL("/login", origin)
    url.searchParams.set("callbackUrl", pathname + search)
    url.searchParams.set("reason", "no-session")
    return NextResponse.redirect(url)
  }

  // Optional OWNER-only UI gate (keep light; backend enforces true authz)
  // If you want to hide owner areas when no session cookie, keep as-is.
  // Role checks are enforced by the backend/API.
  if (OWNER_PREFIXES.some((p) => pathname.startsWith(p))) {
    // Nothing heavy here to keep bundle small
  }

  return NextResponse.next()
}

// middleware.ts (leave the function as you have it)
export const config = {
  matcher: [
    "/",
    "/profile",
    "/dashboard",
    "/dashboard/:path*",
    "/owner",
    "/owner/:path*",
    "/admin",
    "/admin/:path*",
  ],
};
