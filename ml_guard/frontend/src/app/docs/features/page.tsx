import React from 'react';

export default function CoreFeaturesDoc() {
    return (
        <div className="max-w-3xl">
            <h1 className="text-6xl font-black tracking-tighter mb-8">
                Core <br />
                <span className="text-orange-500 italic">Features.</span>
            </h1>
            
            <section className="space-y-12">
                <div>
                    <h2 className="text-3xl font-black text-white mb-4 uppercase tracking-tight">1. SHAP & Explainability</h2>
                    <p className="text-slate-400 leading-relaxed font-medium">
                        ML Guard provides deep transparency into black-box models using SHapley Additive Explanations. 
                        We compute both global feature importance and local instance-level contributions.
                    </p>
                </div>

                <div>
                    <h2 className="text-3xl font-black text-white mb-4 uppercase tracking-tight">2. Drift Sentinel</h2>
                    <p className="text-slate-400 leading-relaxed font-medium">
                        Real-time monitoring for prediction drift (PSI), statistical shift (KS Test), and 
                        concept decay. The Sentinel agent can be deployed as a sidecar to your inference service.
                    </p>
                </div>

                <div>
                    <h2 className="text-3xl font-black text-white mb-4 uppercase tracking-tight">3. Actuarial Risk Engine</h2>
                    <p className="text-slate-400 leading-relaxed font-medium">
                        Our proprietary scoring system calculates an "Insurance Grade" for your models, 
                        evaluating the probability of failure based on historical reliability and current 
                        drift metrics.
                    </p>
                </div>

                <div className="bg-white/5 border border-white/5 p-8 rounded-[2rem]">
                    <h3 className="text-xl font-black mb-4">Plugin System</h3>
                    <p className="text-sm text-slate-500">
                        ML Guard supports modular plugins for data ingestion (Connectors) and 
                        notifications (Alerts). You can write your own plugins using our standard interface.
                    </p>
                </div>
            </section>
        </div>
    );
}
