"use client";

import React from 'react';
import Link from 'next/link';
import { Shield } from 'lucide-react';

export const Footer = () => {
    return (
        <footer className="bg-[#090A0C] pt-32 pb-12 px-6 border-t border-white/5">
            <div className="max-w-7xl mx-auto">
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-12 mb-20">
                    <div className="col-span-2 lg:col-span-2">
                        <Link href="/" className="flex items-center gap-3 mb-8 group">
                            <div className="w-10 h-10 bg-orange-500 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-500">
                                <Shield className="w-6 h-6 text-black fill-current" />
                            </div>
                            <span className="text-xl font-black text-white tracking-tighter uppercase">
                                ML GUARD
                            </span>
                        </Link>
                        <p className="text-slate-500 text-sm max-w-xs font-medium leading-relaxed">
                            The enterprise standard for ML quality governance and regulatory compliance auditing.
                        </p>
                    </div>

                    <div>
                        <h4 className="text-white font-black text-[10px] uppercase tracking-[0.3em] mb-8">Product</h4>
                        <ul className="space-y-4">
                            {["Quality Gates", "Drift Telemetry", "Bias Analysis", "Pricing"].map(link => (
                                <li key={link}><Link href="#" className="text-slate-500 hover:text-orange-500 transition-colors text-xs font-bold">{link}</Link></li>
                            ))}
                        </ul>
                    </div>

                    <div>
                        <h4 className="text-white font-black text-[10px] uppercase tracking-[0.3em] mb-8">Ecosystem</h4>
                        <ul className="space-y-4">
                            {["Documentation", "Python SDK", "REST API", "Open Source"].map(link => (
                                <li key={link}><Link href="#" className="text-slate-500 hover:text-orange-500 transition-colors text-xs font-bold">{link}</Link></li>
                            ))}
                        </ul>
                    </div>

                    <div>
                        <h4 className="text-white font-black text-[10px] uppercase tracking-[0.3em] mb-8">Legal</h4>
                        <ul className="space-y-4">
                            {["Privacy Policy", "Terms of Service", "EU AI Act Compliance", "SLA"].map(link => (
                                <li key={link}><Link href="#" className="text-slate-500 hover:text-orange-500 transition-colors text-xs font-bold">{link}</Link></li>
                            ))}
                        </ul>
                    </div>
                </div>

                <div className="pt-12 border-t border-white/5 flex flex-col md:flex-row justify-between items-center gap-6">
                    <p className="text-[10px] font-black text-slate-600 uppercase tracking-widest">
                        © 2026 Fireflink AI. All Rights Reserved. Engineered for Security.
                    </p>
                    <div className="flex gap-8">
                        <Link href="/login" className="text-[10px] font-black text-slate-500 uppercase tracking-widest hover:text-orange-500 transition-all">Internal Node Login</Link>
                        <span className="text-[10px] font-black text-slate-700 uppercase tracking-widest">v2.1.0-STABLE</span>
                    </div>
                </div>
            </div>
        </footer>
    );
};
