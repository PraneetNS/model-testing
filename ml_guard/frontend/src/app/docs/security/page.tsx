import React from 'react';

export default function SecurityDoc() {
    return (
        <div className="max-w-3xl">
            <h1 className="text-6xl font-black tracking-tighter mb-8">
                Security <br />
                <span className="text-orange-500 italic">Protocols.</span>
            </h1>
            
            <p className="text-xl text-slate-400 font-medium leading-relaxed mb-12">
                Enterprise-grade protection for your models and data pipelines.
            </p>

            <section className="space-y-12">
                <div>
                    <h2 className="text-2xl font-black text-white mb-4 uppercase">Credential Masking</h2>
                    <p className="text-slate-400">
                        ML Guard uses Fernet symmetric encryption for all data connector credentials. 
                        Sensative keys are never logged or stored in plain text.
                    </p>
                </div>

                <div>
                    <h2 className="text-2xl font-black text-white mb-4 uppercase">RBAC & API Keys</h2>
                    <p className="text-slate-400">
                        Access is managed via hashed API keys. We implement granular role-based 
                        access control to ensure only authorized users can trigger audits or 
                        view sensitive risk scores.
                    </p>
                </div>

                <div className="bg-red-500/10 border-l-4 border-red-500 p-8 rounded-r-3xl">
                    <h3 className="text-red-500 font-black uppercase tracking-widest text-sm mb-2">Internal Audit</h3>
                    <p className="text-slate-300 text-sm">
                        All state-changing operations are logged in a tamper-proof security audit log 
                        for compliance reporting (SOC2/ISO27001).
                    </p>
                </div>
            </section>
        </div>
    );
}
