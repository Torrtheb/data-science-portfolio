// src/app/owner/scheduling/ui/components/Tabs.tsx
"use client";
import React from "react";

export function TabButton({ active, onClick, children }:{
  active: boolean; onClick: () => void; children: React.ReactNode;
}) {
  return (
    <button
      className={`px-3 py-2 border rounded-xl ${active ? "bg-black text-white" : "bg-white"}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

export function SubTabButton({ active, onClick, children }:{
  active: boolean; onClick: () => void; children: React.ReactNode;
}) {
  return (
    <button
      className={`px-3 py-2 rounded-lg transition-colors ${active ? "bg-blue-100 text-blue-700 font-medium" : "text-gray-600 hover:text-gray-900 hover:bg-gray-50"}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
}
