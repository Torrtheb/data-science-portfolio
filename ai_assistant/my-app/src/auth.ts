// src/auth.ts
import NextAuth from "next-auth"
import Google from "next-auth/providers/google"
import Credentials from "next-auth/providers/credentials"
import { PrismaAdapter } from "@auth/prisma-adapter"
import { PrismaClient } from "@prisma/client"
import { z } from "zod"
import bcrypt from "bcryptjs"

// --- Prisma client (avoid leaks on dev hot reload)
const g = globalThis as unknown as { prisma?: PrismaClient }
export const prisma =
  g.prisma ??
  new PrismaClient({
    log: process.env.NODE_ENV === "development" ? ["error", "warn"] : ["error"],
  })
if (process.env.NODE_ENV === "development") g.prisma = prisma

type TokenWithExtras = {
  [key: string]: unknown
  sub?: string
  name?: string | null
  email?: string | null
  role?: "OWNER" | "CLIENT"
  timezone?: string
}

export const {
  handlers, // /api/auth
  auth,
  signIn,
  signOut,
} = NextAuth({
  // Ensure NextAuth uses the same secret as the proxy signer
  secret: process.env.NEXTAUTH_SECRET || process.env.BACKEND_NEXTAUTH_SECRET,
  adapter: PrismaAdapter(prisma),
  trustHost: true,
  session: { strategy: "jwt" },
  pages: { signIn: "/login" },

  providers: [
    Google({
      clientId: process.env.GOOGLE_ID!,
      clientSecret: process.env.GOOGLE_SECRET!,
      allowDangerousEmailAccountLinking: true,
      authorization: { params: { prompt: "select_account" } },
    }),
    Credentials({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(raw) {
        // 1) Dev override (DISABLED in production): allow logging in as OWNER without touching Prisma DB
        // Guarded by ALLOW_DEV_LOGIN=1 and NODE_ENV!=='production'
        const devCode = process.env.DEV_LOGIN_CODE;
        const devOwnerId = process.env.DEV_OWNER_ID;
        const allowDevOverride = process.env.ALLOW_DEV_LOGIN === "1" && process.env.NODE_ENV !== "production";
        const rawAny = raw as { email?: string; password?: string } | undefined;
        if (allowDevOverride && devCode && devOwnerId && rawAny?.password === devCode) {
          return {
            id: devOwnerId,
            email: (rawAny.email || process.env.OWNER_EMAIL || "owner@example.com") as string,
            name: "Dev Owner",
            // Mark as OWNER so downstream JWT/session carries role
            role: "OWNER" as const,
            timezone: "America/Toronto",
          } as any;
        }

        // 2) Normal credentials flow against Prisma Users (with hashed password)
        const parsed = z
          .object({ email: z.string().email(), password: z.string().min(8) })
          .safeParse(raw)
        if (!parsed.success) return null

        const { email, password } = parsed.data
        const user = await prisma.user.findUnique({ where: { email } })
        if (!user || !user.password) return null

        const ok = await bcrypt.compare(password, user.password)
        if (!ok) return null

        return {
          id: user.id,
          name: user.name ?? undefined,
          email: user.email!,
          image: user.image ?? undefined,
        }
      },
    }),
  ],

  callbacks: {
    async signIn({ user, account }) {
      // For OAuth (Google), only allow if this email is already provisioned in Prisma.
      if (account?.provider === "google") {
        const dbUser = await prisma.user.findUnique({
          where: { email: (user.email ?? "").toLowerCase() },
          select: { id: true, emailVerified: true, timezone: true },
        });

        // If not provisioned, block sign-in
        if (!dbUser) return "/login?error=NoAccess";
        // Optional: require email verification even for Google (up to you)
        // if (!dbUser.emailVerified) return "/login?error=Verification";
        return true;
      }

      // For credentials, you already verify password in authorize()
      // (Optional) still require verification for credentials logins:
      if (account?.provider === "credentials") {
        const dbUser = await prisma.user.findUnique({
          where: { id: (user as { id?: string } | null | undefined)?.id as string },
          select: { emailVerified: true },
        });
        const requireVerified = (process.env.REQUIRE_VERIFIED_CREDENTIALS ?? "1") !== "0";
        if (requireVerified && !dbUser?.emailVerified) return "/login?error=Verification";
      }

      return true;
    },


    async jwt({ token, user, trigger, session }) {
      const t = token as TokenWithExtras;

      // NEW: allow client-side `useSession().update({ timezone })` to stick
      const sessionUpdate = session as { timezone?: string } | null;
      if (trigger === "update" && sessionUpdate?.timezone) {
        t.timezone = sessionUpdate.timezone;
      }

      // Existing: on login, pull from DB
      if (user) {
        t.sub = (user as { id?: string } | null | undefined)?.id as string;
        t.email = user.email ?? t.email;
        t.name = user.name ?? t.name;

        if (t.email) {
          const dbUser = await prisma.user.findUnique({
            where: { email: (t.email as string).toLowerCase() },
            select: { timezone: true },
          });
          if (dbUser?.timezone) t.timezone = dbUser.timezone;
        }
      }

      if (!t.timezone) t.timezone = "America/Toronto";


      // Role calc (keep your existing logic, but preserve explicit role from 'user')
      const emailLower = ((t.email as string) || "").trim().toLowerCase();
      const OWNER_EMAIL = (process.env.OWNER_EMAIL ?? "")
        .split(",")
        .map((s) => s.trim().toLowerCase())
        .filter(Boolean);
      const FORCE_OWNER_EMAIL = (process.env.FORCE_OWNER_EMAIL ?? "")
        .trim()
        .toLowerCase();
      // If a role was injected by authorize (e.g., dev override), keep it.
      if (!t.role) {
        const isOwner =
          (!!emailLower && OWNER_EMAIL.includes(emailLower)) ||
          (!!FORCE_OWNER_EMAIL && emailLower === FORCE_OWNER_EMAIL);
        t.role = isOwner ? "OWNER" : "CLIENT";
      }
      return t;
    },

    async session({ session, token }) {
      const t = token as TokenWithExtras;
      if (session.user) {
        const userOut = session.user as unknown as {
          id?: string;
          role?: "OWNER" | "CLIENT";
          timezone?: string;
        } & typeof session.user;
        userOut.id = (t.sub as string) ?? userOut.id;
        userOut.role = (t.role as "OWNER" | "CLIENT") ?? "CLIENT";
        userOut.timezone = (t.timezone as string) ?? "America/Toronto";
        session.user.name = (t.name as string) ?? session.user.name;
        session.user.email = (t.email as string) ?? session.user.email;
      }
      return session;
    },
  },
})
