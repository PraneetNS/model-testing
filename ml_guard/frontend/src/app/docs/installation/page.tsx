import React from 'react';

export default function InstallationPage() {
    return (
        <div className="max-w-3xl">
            <h1 className="text-6xl font-black tracking-tighter mb-8">
                Installation <br />
                <span className="text-orange-500 italic">Guide.</span>
            </h1>
            
            <p className="text-xl text-slate-400 font-medium leading-relaxed mb-12">
                Deploy ML Guard in your infrastructure in minutes. We support native 
                Python installation, Docker, and Kubernetes.
            </p>

            <section className="space-y-8 mb-16">
                <h2 className="text-3xl font-black tracking-tight">1. Python SDK</h2>
                <p className="text-slate-400">
                    The easiest way to get started is by installing our core SDK via pip. 
                    This includes the CLI and all necessary scientific libraries.
                </p>
                <div className="bg-[#0D0E12] border border-white/5 p-6 rounded-2xl font-mono text-sm group relative">
                    <pre className="text-orange-500">
                        pip install ml-guard-sdk
                    </pre>
                    <div className="absolute top-4 right-4 text-[10px] text-slate-600 font-bold uppercase tracking-widest">Bash</div>
                </div>
            </section>

            <section className="space-y-8 mb-16">
                <h2 className="text-3xl font-black tracking-tight">2. Backend Deployment</h2>
                <p className="text-slate-400">
                    For enterprise environments, we recommend deploying the full governance platform 
                    using Docker Compose.
                </p>
                <div className="bg-[#0D0E12] border border-white/5 p-6 rounded-2xl font-mono text-sm group relative overflow-x-auto">
                    <pre className="text-slate-300">
{`# Clone the repository
git clone https://github.com/Fireflink/ml_guard.git

# Start the services
cd ml_guard
docker-compose up -d`}
                    </pre>
                </div>
            </section>

            <section className="space-y-8 mb-16">
                <h2 className="text-3xl font-black tracking-tight">3. Environment Setup</h2>
                <p className="text-slate-400">
                    Configure your backend by setting up the <code className="text-orange-500">.env</code> file. 
                    Ensure you have your PostgreSQL and Redis connection strings ready.
                </p>
                <div className="bg-white/5 border border-white/5 p-6 rounded-2xl">
                    <ul className="space-y-4 text-slate-400">
                        <li className="flex items-center gap-3">
                            <span className="w-1.5 h-1.5 bg-orange-500 rounded-full" />
                            <span>DATABASE_URL: Your Postgres connection</span>
                        </li>
                        <li className="flex items-center gap-3">
                            <span className="w-1.5 h-1.5 bg-orange-500 rounded-full" />
                            <span>REDIS_URL: Your Redis bus for Celery tasks</span>
                        </li>
                        <li className="flex items-center gap-3">
                            <span className="w-1.5 h-1.5 bg-orange-500 rounded-full" />
                            <span>MINIO_ENDPOINT: For model artifact storage</span>
                        </li>
                    </ul>
                </div>
            </section>
        </div>
    );
}
