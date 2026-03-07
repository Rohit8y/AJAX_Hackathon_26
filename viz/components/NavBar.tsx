"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "3D Viewer" },
  { href: "/kinematics", label: "WhipChain Analysis" },
];

export default function NavBar() {
  const pathname = usePathname();

  return (
    <header className="flex items-center gap-6 px-5 py-2.5 bg-gray-950/80 backdrop-blur-xl border-b border-gray-800/60 shrink-0"
      style={{ borderImage: "linear-gradient(to right, transparent, #D2001A33, transparent) 1" }}
    >
      <Link href="/" className="text-red-600 font-bold text-lg tracking-wide flex items-center gap-1.5 hover:text-red-500 transition-colors">
        <span>⚽</span>
        <span>AJAX 3D</span>
      </Link>
      <nav className="flex gap-1">
        {LINKS.map(({ href, label }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={`px-3 py-1.5 rounded text-sm transition-colors ${
                active
                  ? "bg-red-900/50 text-red-400 font-medium"
                  : "text-gray-400 hover:text-gray-200 hover:bg-gray-800/60"
              }`}
            >
              {label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
