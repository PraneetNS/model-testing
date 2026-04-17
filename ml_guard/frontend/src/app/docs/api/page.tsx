import React from 'react';

export default function APIReferenceDoc() {
    return (
        <div className="max-w-3xl">
            <h1 className="text-6xl font-black tracking-tighter mb-8">
                API <br />
                <span className="text-orange-500 italic">Reference.</span>
            </h1>
            
            <p className="text-xl text-slate-400 font-medium leading-relaxed mb-12">
                Programmatic interface for the ML Guard Governance Engine.
            </p>

            <div className="space-y-8">
                {[
                    { method: "POST", path: "/api/v1/audit/run", desc: "Trigger a full governance audit on a model and dataset." },
                    { method: "GET", path: "/api/v1/insurance/score", desc: "Retrieve the actuarial risk grade and insurance metrics." },
                    { method: "GET", path: "/api/v1/drift/summary", desc: "Get aggregated drift statistics for a specific model ID." },
                    { method: "POST", path: "/api/v1/explain/shap", desc: "Start an asynchronous SHAP computation job." },
                    { method: "POST", path: "/api/v1/contact", desc: "Send a technical query to the project maintainers." }
                ].map((api, i) => (
                    <div key={i} className="bg-white/5 border border-white/5 p-8 rounded-3xl group">
                        <div className="flex items-center gap-4 mb-4">
                            <span className="px-3 py-1 bg-orange-500 text-black text-[10px] font-black rounded-lg">{api.method}</span>
                            <code className="text-white font-black group-hover:text-orange-500 transition-colors">{api.path}</code>
                        </div>
                        <p className="text-slate-500 text-sm font-medium">{api.desc}</p>
                    </div>
                ))}
            </div>

            <p className="mt-16 text-slate-400 text-sm">
                For the full interactive OpenAPI specification, visit <code className="text-orange-500">http://localhost:8000/docs</code> while the backend is running.
            </p>
        </div>
    );
}
