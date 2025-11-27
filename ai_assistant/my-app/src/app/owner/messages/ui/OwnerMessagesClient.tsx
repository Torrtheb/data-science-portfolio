"use client";

import { SessionProvider } from "next-auth/react";
import BroadcastEmailPanel from "@/app/owner/scheduling/ui/panels/BroadcastEmailPanel";

export default function OwnerMessagesClient() {
  return (
    <SessionProvider>
      <div className="p-6 space-y-6">
        <header className="space-y-1">
          <h1 className="text-2xl font-semibold">Messaging</h1>
          <p className="text-sm text-gray-600">Send updates to clients directly from here.</p>
        </header>
        <BroadcastEmailPanel />
      </div>
    </SessionProvider>
  );
}
