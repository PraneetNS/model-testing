import React from 'react';

export default function DocsPage() {
    return (
        <div className="max-w-3xl">
            <h1 className="text-6xl font-black tracking-tighter mb-8">
                Documentation <br />
                <span className="text-orange-500 italic">Overview.</span>
            </h1>
            
            <p className="text-xl text-slate-400 font-medium leading-relaxed mb-12">
                ML Guard is an enterprise-grade platform designed to bring transparency, 
                compliance, and operational stability to your AI models.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-16">
                {[
                    { title: "Get Started", desc: "Setting up your first governance project", link: "/docs/installation" },
                    { title: "Architecture", desc: "Deep dive into the engine and core logic", link: "/docs/features" },
                    { title: "Developers", desc: "Python SDK and CLI integration guides", link: "/docs/sdk" },
                    { title: "REST API", desc: "Full documentation for backend integration", link: "/docs/api" },
                ].map((card, i) => (
                    <div key={i} className="bg-white/5 border border-white/5 p-8 rounded-3xl hover:border-orange-500/30 transition-all cursor-pointer group">
                        <h3 className="text-xl font-black mb-2 group-hover:text-orange-500 transition-colors uppercase tracking-tight">{card.title}</h3>
                        <p className="text-slate-400 text-sm">{card.desc}</p>
                    </div>
                ))}
            </div>

            <section className="space-y-8">
                <h2 className="text-3xl font-black tracking-tight">Technical Governance</h2>
                <p className="text-slate-400">
                    ML Guard doesn't just monitor; it provides a deterministic framework for model quality. 
                    Unlike traditional monitoring tools that give you "maybe" alerts, ML Guard uses precise statistical 
                    gates like PSI, KS-test, and Actuarial Risk Scoring to ensure your models are always compliant.
                </p>
                <div className="bg-orange-500/10 border-l-4 border-orange-500 p-6 rounded-r-2xl">
                    <p className="text-orange-500 font-black text-sm uppercase tracking-widest mb-2">Notice</p>
                    <p className="text-slate-300 text-sm">
                        This documentation covers ML Guard v8.2 Enterprise. Some features like SHAP-based 
                        fairness tracking and data connectors require a valid API key.
                    </p>
                </div>
            </section>
        </div>
    );
}
