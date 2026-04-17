"use client";

import React from 'react';
import { Navbar } from '@/components/landing/Navbar';
import { Footer } from '@/components/landing/Footer';
import { motion } from 'framer-motion';
import { 
    Zap, Activity, Scale, Lock, FileCheck, GitBranch, 
    Eye, ShieldCheck, Database, Layout, Webhook, Cpu 
} from 'lucide-react';

const allFeatures = [
    {
        title: "Model Audit (Classical ML)",
        desc: "Deep statistical analysis for tabular and classical models. Tracking Accuracy, F1, PSI/KS Drift, Overfitting, and Calibration data.",
        icon: Activity,
        category: "Governance"
    },
    {
        title: "SHAP Explainability",
        desc: "Local and global feature importance using SHapley values. Identify fairness-drift correlations and protected attribute impact.",
        icon: Eye,
        category: "Transparency"
    },
    {
        title: "Actuarial Insurance Scoring",
        desc: "Standardized risk grades (A++ to F) for enterprise AI. Actuarial modeling for reliability, robustness, and compliance.",
        icon: ShieldCheck,
        category: "Risk"
    },
    {
        title: "RAG & LLM Observability",
        desc: "Monitoring GenAI grounding, retrieval hit rates, and context relevance for Large Language Model pipelines.",
        icon: Zap,
        category: "GenAI"
    },
    {
        title: "Enterprise Data Connectors",
        desc: "Direct ingestion from S3, GCS, Snowflake, and BigQuery. Secure modular plugin system for enterprise data sources.",
        icon: Database,
        category: "Connectors"
    },
    {
        title: "CI/CD Sync Gates",
        desc: "Deterministic quality gates that stop low-quality models in GitHub Actions or Jenkins before they hit production.",
        icon: GitBranch,
        category: "Automation"
    },
    {
        title: "Sentinel Real-time Monitor",
        desc: "A lightweight agent that monitors live prediction streams for adversarial attacks and concept decay.",
        icon: Lock,
        category: "Security"
    },
    {
        title: "Notification Plugins",
        desc: "Real-time alerts via Slack and Microsoft Teams when model degradation or security breaches are detected.",
        icon: Webhook,
        category: "Plugins"
    },
    {
        title: "Comprehensive SDK",
        desc: "Python SDK and CLI for seamless integration into any data science workflow or ML platform.",
        icon: Cpu,
        category: "Developer"
    }
];

export default function FeaturesPage() {
    return (
        <main className="bg-[#090A0C] min-h-screen">
            <Navbar />
            
            <div className="pt-40 pb-20 px-6">
                <div className="max-w-7xl mx-auto">
                    <motion.div 
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="text-center mb-24"
                    >
                        <h1 className="text-5xl md:text-7xl font-black text-white tracking-tighter mb-6">
                            Enterprise <span className="text-orange-500 italic">Capabilities.</span>
                        </h1>
                        <p className="text-slate-400 text-xl max-w-2xl mx-auto font-medium lead-relaxed">
                            Discover the modules that make ML Guard the standard for 
                            technical AI governance and operational intelligence.
                        </p>
                    </motion.div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                        {allFeatures.map((f, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, y: 30 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: i * 0.05 }}
                                className="bg-[#111318] border border-white/5 p-12 rounded-[3rem] hover:border-orange-500/30 transition-all group"
                            >
                                <div className="w-16 h-16 bg-orange-500/10 rounded-2xl flex items-center justify-center mb-10 group-hover:scale-110 transition-transform">
                                    <f.icon className="w-8 h-8 text-orange-500" />
                                </div>
                                <span className="text-[10px] font-black text-slate-500 uppercase tracking-[0.3em] mb-4 block">
                                    {f.category}
                                </span>
                                <h3 className="text-2xl font-black text-white mb-4 tracking-tight group-hover:text-orange-500 transition-colors">
                                    {f.title}
                                </h3>
                                <p className="text-slate-400 leading-relaxed font-medium">
                                    {f.desc}
                                </p>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </div>

            <Footer />
        </main>
    );
}
