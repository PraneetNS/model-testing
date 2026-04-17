import React from 'react';

export default function SDKDoc() {
    return (
        <div className="max-w-3xl">
            <h1 className="text-6xl font-black tracking-tighter mb-8">
                SDK & <br />
                <span className="text-orange-500 italic">CLI.</span>
            </h1>
            
            <p className="text-xl text-slate-400 font-medium leading-relaxed mb-12">
                Integrating ML Guard into your Python workflow or terminal pipelines.
            </p>

            <section className="space-y-12 mb-20">
                <div>
                    <h2 className="text-2xl font-black text-white mb-4 uppercase">Python SDK</h2>
                    <div className="bg-[#0D0E12] border border-white/5 p-6 rounded-2xl font-mono text-xs text-orange-500 overflow-x-auto">
                        <pre>
{`import ml_guard as mlg

# Initialize client
client = mlg.Client(api_url="http://localhost:8000")

# Run a quick audit
report = client.audit(
    model=my_model,
    test_data=df_test,
    target_col="prediction"
)

print(f"Governance Score: {report.score}")`}
                        </pre>
                    </div>
                </div>

                <div>
                    <h2 className="text-2xl font-black text-white mb-4 uppercase">Command Line</h2>
                    <div className="bg-[#0D0E12] border border-white/5 p-6 rounded-2xl font-mono text-xs text-slate-300 overflow-x-auto">
                        <pre>
{`$ ml-guard scan --model ./model.joblib --data ./prod_data.csv
[INFO] Connecting to ML Guard Engine...
[INFO] Analysis in progress [############] 100%
[SUCCESS] Audit Complete. Score: 92/100 (Grade: A)`}
                        </pre>
                    </div>
                </div>
            </section>
        </div>
    );
}
