"use client";

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Navbar } from '@/components/landing/Navbar';
import { cn } from '@/lib/utils';
import { Book, Code, Rocket, Terminal, Layers, ShieldCheck } from 'lucide-react';

const sidebarLinks = [
    { title: "Introduction", href: "/docs", icon: Book },
    { title: "Installation", href: "/docs/installation", icon: Rocket },
    { title: "Core Features", href: "/docs/features", icon: Layers },
    { title: "SDK & CLI", href: "/docs/sdk", icon: Terminal },
    { title: "API Reference", href: "/docs/api", icon: Code },
    { title: "Security Protocols", href: "/docs/security", icon: ShieldCheck },
];

export default function DocsLayout({ children }: { children: React.ReactNode }) {
    const pathname = usePathname();

    return (
        <main className="bg-[#090A0C] min-h-screen text-white">
            <Navbar />
            
            <div className="pt-24 max-w-7xl mx-auto px-6 flex gap-12">
                {/* Sidebar */}
                <aside className="w-64 flex-shrink-0 hidden lg:block sticky top-24 h-[calc(100vh-6rem)] overflow-y-auto pt-12">
                    <div className="space-y-2">
                        {sidebarLinks.map((link) => (
                            <Link
                                key={link.href}
                                href={link.href}
                                className={cn(
                                    "flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 group",
                                    pathname === link.href 
                                        ? "bg-orange-500 text-black font-black" 
                                        : "text-slate-400 hover:text-white hover:bg-white/5"
                                )}
                            >
                                <link.icon className={cn("w-4 h-4", pathname === link.href ? "text-black" : "text-slate-500 group-hover:text-orange-500")} />
                                <span className="text-[11px] uppercase tracking-widest">{link.title}</span>
                            </Link>
                        ))}
                    </div>
                </aside>

                {/* Content */}
                <div className="flex-1 py-12 prose prose-invert prose-orange max-w-none">
                    {children}
                </div>
            </div>
        </main>
    );
}
