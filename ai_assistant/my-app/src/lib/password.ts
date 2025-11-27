// lib/password.ts
import { z } from "zod";

export const passwordSchema = z
  .string()
  .min(10, "Password must be at least 10 characters")
  .max(64, "Password must be at most 64 characters")
  .regex(/[a-z]/, "Password must include a lowercase letter")
  .regex(/[A-Z]/, "Password must include an uppercase letter")
  .regex(/[0-9]/, "Password must include a digit")
  .regex(/[^A-Za-z0-9]/, "Password must include a special character")
  .refine((v) => v.trim() === v, "Password must not have leading or trailing spaces")
  // bcrypt cap is 72 bytes (not chars)
  .refine((v) => Buffer.byteLength(v, "utf8") <= 72, "Password is too long for bcrypt");
