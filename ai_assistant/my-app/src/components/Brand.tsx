"use client"

import Image from "next/image"
import Link from "next/link"

const STUDIO_NAME = process.env.NEXT_PUBLIC_STUDIO_NAME || "Your Music Studio"

export default function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <Link href="/" className="flex items-center gap-2 select-none group">
      <Image
        src="/logo-studio.svg"
        alt="Studio logo"
        width={compact ? 30 : 36}
        height={compact ? 30 : 36}
        priority
      />
      <span
        className={`font-semibold tracking-tight text-[color:rgb(var(--brand))] group-hover:opacity-90 transition-opacity ${
          compact ? "text-base" : "text-lg"
        }`}
        style={{
          // Fallback if CSS var is missing
          color: "rgb(var(--brand))",
        }}
      >
        {STUDIO_NAME}
      </span>
    </Link>
  )
}
