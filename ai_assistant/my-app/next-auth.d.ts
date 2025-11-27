import "next-auth"

declare module "next-auth" {
  interface Session {
    user: {
      id: string
      role: "OWNER" | "STAFF" | "CLIENT"
      timezone: string
      name?: string | null
      email?: string | null
      image?: string | null
    }
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    role?: "OWNER" | "STAFF" | "CLIENT"
    timezone?: string
  }
}
