"use client";

import React from 'react';
import { Navbar } from '@/components/landing/Navbar';
import { Footer } from '@/components/landing/Footer';
import { motion } from 'framer-motion';
import { Terminal, Cpu, Database, ShieldCheck, Box, Workflow } from 'lucide-react';

const steps = [
    {
        title: "Model Onboarding",
        content: "Integrate your models via our Python SDK or connect directly to your registry (MLflow, Hugging Face). We support Sklearn, PyTorch, and TensorFlow natively.",
        icon: Box
    },
    {
        title: "Data Ingestion",
        content: "Pull validation or production datasets using our Enterprise Connectors (S3, Snowflake, BigQuery) to establish a baseline for your model's performance.",
        icon: Database
    },
    {
        title: "Governance Analysis",
        content: "The ML Guard Engine runs deep statistical audits: PSI drift, equity scans, SHAP transparency, and actuarial risk tiering occur in parallel.",
        icon: Cpu
    },
    {
        title: "CI/CD & Gates",
        content: "Embed ML Guard in your pipelines. If a model fails to meet your governance policies, the Sync Gate prevents deployment automatically.",
        icon: Workflow
    },
    {
        title: "Real-time Sentinel",
        content: "Deploy our lightweight agent alongside your model to monitor live traffic for drift, bias, and adversarial attacks as they happen.",
        icon: ShieldCheck
    }
];

export default function HowItWorksPage() {
    return (
        <main className="bg-[#090A0C] min-h-screen">
            <Navbar />
            
            <div className="pt-40 pb-20 px-6">
                <div className="max-w-7xl mx-auto">
                    <motion.div 
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="text-center mb-32"
                    >
                        <h1 className="text-5xl md:text-7xl font-black text-white tracking-tighter mb-6">
                            Deterministic <span className="text-orange-500 italic">Workflow.</span>
                        </h1>
                        <p className="text-slate-400 text-xl max-w-2xl mx-auto font-medium leading-relaxed">
                            A seamless bridge from data science experimentation to enterprise-grade production governance.
                        </p>
                    </motion.div>

                    <div className="space-y-32">
                        {steps.map((step, i) => (
                            <motion.div 
                                key={i}
                                initial={{ opacity: 0, x: i % 2 === 0 ? -50 : 50 }}
                                whileInView={{ opacity: 1, x: 0 }}
                                viewport={{ once: true }}
                                className={`flex flex-col ${i % 2 === 0 ? 'md:flex-row' : 'md:flex-row-reverse'} items-center gap-20`}
                            >
                                <div className="flex-1">
                                    <div className="w-16 h-16 bg-white/5 rounded-2xl flex items-center justify-center mb-8 border border-white/5">
                                        <step.icon className="w-8 h-8 text-orange-500" />
                                    </div>
                                    <h2 className="text-4xl font-black text-white mb-6 uppercase tracking-tight">
                                        {i + 1}. {step.title}
                                    </h2>
                                    <p className="text-slate-400 text-lg leading-relaxed font-medium">
                                        {step.content}
                                    </p>
                                </div>
                                <div className="flex-1 w-full bg-[#111318] aspect-video rounded-[3rem] border border-white/5 relative overflow-hidden flex items-center justify-center">
                                     <div className="absolute inset-0 bg-gradient-to-br from-orange-500/10 to-transparent" />
                                     <div className="z-10 text-6xl font-black text-orange-500/20">
                                         0{i + 1}
                                     </div>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </div>

            <Footer />
        </main>
    );
}
