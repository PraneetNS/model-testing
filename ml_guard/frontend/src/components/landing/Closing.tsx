"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Star, ChevronDown, Send, Github, Twitter, Linkedin } from 'lucide-react';
import { cn } from '@/lib/utils';

export const Testimonials = () => {
    const reviews = [
        {
            name: "Sarah Chen",
            role: "Head of AI, Fintech Global",
            text: "ML Guard revolutionized our model deployment cycle. We went from weeks of manual audit to automated quality gates in hours.",
            rating: 5
        },
        {
            name: "Marcus Thorne",
            role: "Chief Compliance Officer, MediLog",
            text: "The regulatory reporting features are unparalleled. Finally, a platform that speaks the language of both engineers and auditors.",
            rating: 5
        },
        {
            name: "Elena Rodriguez",
            role: "MLOps Architect, CloudScale",
            text: "The drift telemetry is insanely detailed. We caught a critical concept decay event before it impacted our revenue by 10%.",
            rating: 5
        }
    ];

    return (
        <section className="py-32 bg-[#090A0C]">
            <div className="max-w-7xl mx-auto px-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {reviews.map((r, i) => (
                        <motion.div
                            key={i}
                            initial={{ opacity: 0, scale: 0.9 }}
                            whileInView={{ opacity: 1, scale: 1 }}
                            viewport={{ once: true }}
                            transition={{ delay: i * 0.1 }}
                            className="bg-white/[0.02] border border-white/5 p-10 rounded-[2.5rem] relative"
                        >
                            <div className="flex gap-1 mb-6">
                                {[...Array(r.rating)].map((_, i) => (
                                    <Star key={i} className="w-4 h-4 text-orange-500 fill-current" />
                                ))}
                            </div>
                            <p className="text-white font-medium mb-8 leading-relaxed italic">"{r.text}"</p>
                            <div>
                                <h4 className="text-white font-black text-sm">{r.name}</h4>
                                <p className="text-slate-500 text-[10px] uppercase font-black tracking-widest mt-1">{r.role}</p>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
};

export const Docs = () => {
    const [openIndex, setOpenIndex] = useState(0);

    const items = [
        { title: "Installation Guide", content: "npm install @ml-guard/sdk-core" },
        { title: "CLI usage", content: "ml-guard scan --model ./model.pkl --train ./data.csv" },
        { title: "Python SDK usage", content: "import ml_guard as mlg\nclient = mlg.Client(token=...)" },
        { title: "CI/CD example", content: "uses: fireflink/ml-guard-action@v2\nwith:\n  project-id: ${{ secrets.ML_GATE_ID }}" }
    ];

    return (
        <section id="docs" className="py-32 bg-[#090A0C]">
            <div className="max-w-3xl mx-auto px-6">
                <h2 className="text-4xl font-black text-white text-center mb-16 tracking-tighter italic">Developer Protocol.</h2>
                <div className="space-y-4">
                    {items.map((item, i) => (
                        <div key={i} className="bg-white/[0.02] border border-white/5 rounded-2xl overflow-hidden">
                            <button
                                onClick={() => setOpenIndex(openIndex === i ? -1 : i)}
                                className="w-full px-8 py-6 flex items-center justify-between text-left hover:bg-white/[0.04] transition-all"
                            >
                                <span className="text-sm font-black text-white uppercase tracking-widest">{item.title}</span>
                                <ChevronDown className={cn("w-5 h-5 transition-transform", openIndex === i ? "rotate-180" : "")} />
                            </button>
                            <AnimatePresence>
                                {openIndex === i && (
                                    <motion.div
                                        initial={{ height: 0 }}
                                        animate={{ height: 'auto' }}
                                        exit={{ height: 0 }}
                                        className="overflow-hidden"
                                    >
                                        <div className="px-8 pb-8">
                                            <pre className="bg-black/40 p-6 rounded-xl border border-white/5 text-orange-500 font-mono text-xs overflow-x-auto">
                                                <code>{item.content}</code>
                                            </pre>
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
};

export const Contact = () => {
    return (
        <section className="py-32 bg-[#090A0C]">
            <div className="max-w-7xl mx-auto px-6">
                <div className="bg-gradient-to-br from-[#111318] to-black border border-white/5 rounded-[4rem] p-12 md:p-20 grid grid-cols-1 lg:grid-cols-2 gap-20 shadow-2xl">
                    <div>
                        <h2 className="text-5xl font-black text-white tracking-tighter mb-8">Let's Secure <br /> Your AI Fleet.</h2>
                        <p className="text-slate-400 font-medium mb-12">Talk to our solutions architects about integrating ML Guard into your production ecosystem.</p>

                        <div className="flex gap-6">
                            {[Github, Twitter, Linkedin].map((Icon, i) => (
                                <button key={i} className="w-12 h-12 bg-white/5 border border-white/5 rounded-xl flex items-center justify-center hover:bg-orange-500 hover:text-black transition-all">
                                    <Icon className="w-5 h-5" />
                                </button>
                            ))}
                        </div>
                    </div>

                    <form className="space-y-6">
                        <div className="grid grid-cols-2 gap-6">
                            <input className="bg-white/5 border border-white/5 rounded-2xl px-6 py-4 text-white focus:outline-none focus:border-orange-500/50 transition-all font-bold text-sm" placeholder="Full Name" />
                            <input className="bg-white/5 border border-white/5 rounded-2xl px-6 py-4 text-white focus:outline-none focus:border-orange-500/50 transition-all font-bold text-sm" placeholder="Work Email" />
                        </div>
                        <input className="w-full bg-white/5 border border-white/5 rounded-2xl px-6 py-4 text-white focus:outline-none focus:border-orange-500/50 transition-all font-bold text-sm" placeholder="Company" />
                        <textarea className="w-full bg-white/5 border border-white/5 rounded-2xl px-6 py-8 text-white focus:outline-none focus:border-orange-500/50 transition-all font-bold text-sm h-32" placeholder="Tell us about your requirements..." />
                        <button className="w-full bg-orange-500 text-black py-5 rounded-2xl font-black uppercase tracking-widest hover:bg-orange-400 transition-all flex items-center justify-center gap-3">
                            Transmit Signal <Send className="w-4 h-4" />
                        </button>
                    </form>
                </div>
            </div>
        </section>
    );
};
