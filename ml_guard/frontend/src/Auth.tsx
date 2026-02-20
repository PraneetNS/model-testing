import React, { useState } from 'react';

interface AuthProps {
    onAuthSuccess: (token: string) => void;
}

export const Auth: React.FC<AuthProps> = ({ onAuthSuccess }) => {
    const [isLogin, setIsLogin] = useState(true);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [tenant, setTenant] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        const endpoint = isLogin ? '/auth/login' : '/auth/register';
        const body = isLogin
            ? new URLSearchParams({ username: email, password })
            : JSON.stringify({ email, password, tenant_name: tenant, full_name: email.split('@')[0] });

        try {
            const baseUrl = `http://${window.location.hostname}:8000/api/v1`;
            const response = await fetch(`${baseUrl}${endpoint}`, {
                method: 'POST',
                headers: isLogin ? { 'Content-Type': 'application/x-www-form-urlencoded' } : { 'Content-Type': 'application/json' },
                body: body
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'Authentication failed');

            if (isLogin) {
                onAuthSuccess(data.access_token);
            } else {
                setIsLogin(true);
                alert('Registration successful! Please login.');
            }
        } catch (err: any) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-[#09090B] flex items-center justify-center p-6 font-['Outfit']">
            <div className="w-full max-w-md bg-white/[0.02] border border-white/5 p-10 rounded-[2.5rem] backdrop-blur-3xl shadow-2xl">
                <div className="mb-10 text-center">
                    <span className="bg-orange-500/10 text-orange-500 text-[10px] font-black tracking-[0.3em] uppercase px-4 py-1.5 rounded-full border border-orange-500/20">
                        Enterprise Governance
                    </span>
                    <h1 className="text-4xl font-black text-white mt-6 tracking-tighter">ML GUARD V2</h1>
                    <p className="text-slate-500 text-sm mt-2">{isLogin ? 'Welcome back, Architect.' : 'Create your high-security tenant.'}</p>
                </div>

                <form onSubmit={handleSubmit} className="space-y-4">
                    {!isLogin && (
                        <div className="space-y-1.5">
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest ml-1">Tenant Name</label>
                            <input
                                type="text" value={tenant} onChange={(e) => setTenant(e.target.value)} required
                                className="w-full bg-black/40 border border-white/5 rounded-2xl px-5 py-4 text-white focus:outline-none focus:border-orange-500/50 transition-all placeholder:text-slate-800"
                                placeholder="e.g. Acme Fintech"
                            />
                        </div>
                    )}
                    <div className="space-y-1.5">
                        <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest ml-1">Work Email</label>
                        <input
                            type="email" value={email} onChange={(e) => setEmail(e.target.value)} required
                            className="w-full bg-black/40 border border-white/5 rounded-2xl px-5 py-4 text-white focus:outline-none focus:border-orange-500/50 transition-all placeholder:text-slate-800"
                            placeholder="architect@company.com"
                        />
                    </div>
                    <div className="space-y-1.5">
                        <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest ml-1">Security Key</label>
                        <input
                            type="password" value={password} onChange={(e) => setPassword(e.target.value)} required
                            className="w-full bg-black/40 border border-white/5 rounded-2xl px-5 py-4 text-white focus:outline-none focus:border-orange-500/50 transition-all placeholder:text-slate-800"
                            placeholder="••••••••"
                        />
                    </div>

                    {error && (
                        <div className="bg-red-500/10 border border-red-500/20 p-4 rounded-2xl text-red-500 text-xs font-bold animate-pulse">
                            {error}
                        </div>
                    )}

                    <button
                        disabled={loading}
                        className="w-full bg-white text-black font-black py-5 rounded-2xl hover:bg-orange-500 hover:text-white transition-all transform active:scale-[0.98] disabled:opacity-50 mt-4"
                    >
                        {loading ? 'PROCESSING...' : isLogin ? 'AUTHORIZE ACCESS' : 'INITIALIZE TENANT'}
                    </button>
                </form>

                <div className="mt-8 text-center">
                    <button
                        onClick={() => setIsLogin(!isLogin)}
                        className="text-xs text-slate-500 hover:text-white transition-all font-bold"
                    >
                        {isLogin ? "Don't have a tenant? Request access" : "Already a member? Secure login"}
                    </button>
                </div>
            </div>
        </div>
    );
};
