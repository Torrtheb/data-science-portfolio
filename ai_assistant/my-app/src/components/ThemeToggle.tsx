"use client";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

export default function ThemeToggle() {
  const { theme, systemTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return null;

  const effective = theme === "system" ? systemTheme : theme;
  const next = effective === "dark" ? "light" : "dark";

  const onClick = () => {
    setTheme(next!);
    try { localStorage.setItem("theme", next!); } catch {}
    document.cookie = `theme=${encodeURIComponent(next!)}; Path=/; Max-Age=31536000; SameSite=Lax`;
    document.documentElement.style.colorScheme = next === "dark" ? "dark" : "light";
  };

  return (
    <button
      type="button"
      onClick={onClick}
      className="text-sm px-2 py-1 rounded border hover:bg-gray-50 dark:hover:bg-zinc-800"
      aria-label="Toggle dark mode"
      title="Toggle dark mode"
    >
      {effective === "dark" ? "Light mode" : "Dark mode"}
    </button>
  );
}
