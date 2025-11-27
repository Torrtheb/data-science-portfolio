// src/app/owner/assistant/page.tsx
"use client";
import AssistantChat from "@/components/AssistantChat";
import dynamic from "next/dynamic";

const OutboxListClient = dynamic(() => import("./OutboxListClient"), { ssr: false });

export default function OwnerAssistant() {
  return (
    <div className="space-y-6 p-4">
      <AssistantChat />
      <OutboxListClient />
    </div>
  );
}
