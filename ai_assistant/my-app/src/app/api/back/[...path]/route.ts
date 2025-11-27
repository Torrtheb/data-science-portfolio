// src/app/api/back/[...path]/route.ts
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/auth";
import jwt from "jsonwebtoken";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";
// Use only the configured secrets; avoid implicit placeholders that cause mismatches
const NEXTAUTH_SECRET = process.env.NEXTAUTH_SECRET || process.env.BACKEND_NEXTAUTH_SECRET;

// --- CSRF helpers ---
function parseOrigin(u?: string | null): string | null {
  if (!u) return null;
  try { return new URL(u).origin; } catch { return null; }
}

function collectAllowedOrigins(req: NextRequest): string[] {
  const self = req.nextUrl.origin;
  const envs = [
    process.env.NEXT_PUBLIC_APP_URL,
    process.env.NEXTAUTH_URL,
    process.env.NEXT_PUBLIC_BASE_URL,
    process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : undefined,
  ].filter(Boolean) as string[];
  // Always include self as allowed origin
  const out = new Set<string>([self, ...envs]);
  return Array.from(out);
}

function isAllowedOrigin(candidate: string | null, allowed: string[]): boolean {
  if (!candidate) return false;
  return allowed.includes(candidate);
}

export async function HEAD(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, ctx);
}
export async function GET(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, ctx);
}
export async function POST(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, ctx);
}
export async function PUT(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, ctx);
}
export async function PATCH(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, ctx);
}
export async function DELETE(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return proxy(req, ctx);
}

function _cleanPathSegments(segs: string[] | undefined): string[] {
  if (!Array.isArray(segs)) return [];
  return segs.map((s) =>
    (s || "")
      // normalize smart quotes/ellipsis that can sneak in from copy/paste
      .replace(/\u201c|\u201d/g, '"')
      .replace(/\u2026/g, "...")
      .replace(/["']/g, "")
      // strip anything that isn't path-safe (avoid weird unicode causing 404s)
      .replace(/[^A-Za-z0-9._~-]/g, "")
      .replace(/[.]+$/g, "") // drop trailing dots introduced by ellipsis
      .trim()
  );
}

async function proxy(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const clean = _cleanPathSegments(path);

  // Build target once
  const target = `${BACKEND_URL}/${clean.join("/")}${req.nextUrl.search}`;

  // Enforce CSRF for mutating requests: require same-origin Origin/Referer.
  const method = req.method;
  const isUnsafe = !["GET", "HEAD", "OPTIONS"].includes(method);
  if (isUnsafe) {
    const allowed = collectAllowedOrigins(req);
    const origin = req.headers.get("origin");
    const referer = req.headers.get("referer");
    const originOk = isAllowedOrigin(parseOrigin(origin), allowed) || isAllowedOrigin(parseOrigin(referer), allowed);
    if (!originOk) {
      return new NextResponse(JSON.stringify({ error: "Forbidden (bad origin)" }), { status: 403, headers: { "content-type": "application/json" } });
    }
    // Optional double-submit token (enable by setting REQUIRE_CSRF_FOR_PROXY=1)
    if (process.env.REQUIRE_CSRF_FOR_PROXY === "1") {
      const hdr = req.headers.get("x-csrf-token");
      const cookieTok = req.cookies.get("x-csrf-token")?.value;
      if (!hdr || !cookieTok || hdr !== cookieTok) {
        return new NextResponse(JSON.stringify({ error: "Forbidden (missing/invalid CSRF token)" }), { status: 403, headers: { "content-type": "application/json" } });
      }
    }
  }

  // Get NextAuth session
  const session = await auth();
  const suser = session?.user as ({ id?: string; email?: string; role?: "OWNER" | "CLIENT"; timezone?: string } | undefined);

  // Sign a compact JWT for the backend
  let authHeader: string | undefined;
  if (suser && NEXTAUTH_SECRET) {
    const payload = {
      sub: suser.id as string,
      email: suser.email as string | undefined,
      role: (suser.role ?? "CLIENT") as "OWNER" | "CLIENT",
      timezone: (suser.timezone ?? "America/Toronto") as string,
    };
    const token = jwt.sign(payload, NEXTAUTH_SECRET, { algorithm: "HS256", expiresIn: "10m" });
    authHeader = `Bearer ${token}`;
  }

  // Clone headers (drop cookies and existing Authorization)
  const headers: HeadersInit = {};
  req.headers.forEach((v, k) => {
    const key = k.toLowerCase();
    if (key === "cookie" || key === "authorization") return;
    headers[k] = v;
  });
  if (authHeader) headers["Authorization"] = authHeader;

  // For best SSE behavior: always no-store
  headers["Cache-Control"] = "no-cache";
  // Preserve Accept; EventSource sets "text/event-stream"

  const body = ["GET", "HEAD"].includes(method) ? undefined : await req.arrayBuffer();

  const resp = await fetch(target, {
    method,
    headers,
    body,
    redirect: "manual",
    cache: "no-store",
  });

  // Handle 204 quickly
  if (resp.status === 204) {
    return new NextResponse(null, { status: 204 });
  }

  // Forward response headers but fix for SSE
  const outHeaders = new Headers(resp.headers);
  // Debug-only hint: whether proxy attached Authorization to backend request
  outHeaders.set("x-proxy-auth", authHeader ? "attached" : "missing");
  // Extra debug: surface which backend URL the proxy used and full target
  try {
    outHeaders.set("x-proxy-backend-url", BACKEND_URL);
    outHeaders.set("x-proxy-target", target);
  } catch {}
  outHeaders.delete("set-cookie");

  const ct = outHeaders.get("content-type") || "";
  const isSSE = ct.startsWith("text/event-stream");
  const wantsSSE = (req.headers.get("accept") || "").includes("text/event-stream");

  if (isSSE) {
    // Remove compression so Node won't buffer
    outHeaders.delete("content-encoding");
    outHeaders.set("content-type", "text/event-stream; charset=utf-8");
    outHeaders.set("cache-control", "no-cache, no-transform");
    outHeaders.set("connection", "keep-alive");
  }

  // If the backend rate-limited an SSE request, convert it to a one-shot SSE event
  if (!isSSE && wantsSSE && resp.status === 429) {
    const text = await resp.text().catch(() => "");
    const payload = (() => {
      try { return JSON.parse(text); } catch { return { error: text || "Rate limit exceeded." }; }
    })();
    const msg = typeof payload?.detail === "string" ? payload.detail : (payload?.error || "Rate limit exceeded.");
    const stream = new ReadableStream({
      start(controller) {
        const enc = new TextEncoder();
        const data = JSON.stringify({ message: msg });
        controller.enqueue(enc.encode(`event: rate_limit\n`));
        controller.enqueue(enc.encode(`data: ${data}\n\n`));
        controller.close();
      }
    });
    const sseHeaders = new Headers();
    sseHeaders.set("content-type", "text/event-stream; charset=utf-8");
    sseHeaders.set("cache-control", "no-cache, no-transform");
    sseHeaders.set("connection", "keep-alive");
    return new NextResponse(stream, { status: 200, headers: sseHeaders });
  }

  // IMPORTANT: don't touch resp.body (it's a stream). Just pass through.
  return new NextResponse(resp.body, {
    status: resp.status,
    statusText: resp.statusText,
    headers: outHeaders,
  });
}
