// src/app/api/register/route.ts
import { NextResponse } from "next/server";
export async function POST() {
  return NextResponse.json({ error: "Registration disabled" }, { status: 403 });
}
export async function GET() {
  return NextResponse.json({ error: "Registration disabled" }, { status: 403 });
}
