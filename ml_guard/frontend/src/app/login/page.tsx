"use client";

import React, { useState, useEffect, Suspense } from 'react';
import {
    signInWithEmailAndPassword,
    createUserWithEmailAndPassword,
    signInWithPopup,
    GoogleAuthProvider,
    onAuthStateChanged
} from 'firebase/auth';
import { auth } from '@/lib/firebase';
import { useRouter, useSearchParams } from 'next/navigation';
import { Shield, ChevronLeft, Github } from 'lucide-react';
import Link from 'next/link';
import { motion } from 'framer-motion';

function LoginContent() {
    const searchParams = useSearchParams();
    const router = useRouter();
    const [isLogin, setIsLogin] = useState(!searchParams.get('signup'));
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, (user) => {
            if (user) {
                router.push('/dashboard');
            }
        });
        return () => unsubscribe();
    }, [router]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        try {
            if (isLogin) {
                await signInWithEmailAndPassword(auth, email, password);
            } else {
                await createUserWithEmailAndPassword(auth, email, password);
            }
            router.push('/dashboard');
        } catch (err: any) {
            setError(mapAuthError(err.code));
        } finally {
            setLoading(false);
        }
    };

    const handleGoogleLogin = async () => {
        setLoading(true);
        setError('');
        try {
            const provider = new GoogleAuthProvider();
            await signInWithPopup(auth, provider);
            router.push('/dashboard');
        } catch (err: any) {
            setError(mapAuthError(err.code));
        } finally {
            setLoading(false);
        }
    };

    const mapAuthError = (code: string) => {
        switch (code) {
            case 'auth/user-not-found': return 'Tenant not identified.';
            case 'auth/wrong-password': return 'Invalid access credentials.';
            case 'auth/email-already-in-use': return 'Identity already exists.';
            default: return `Auth Error: ${code.split('/')[1] || 'Unknown'}`;
        }
    };

    return (
        <div className="min-h-screen bg-[#090A0C] flex items-center justify-center p-6 relative overflow-hidden font-['Outfit']">
            {/* Background Effects */}
            <div className="absolute inset-0 z-0">
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-orange-500/5 rounded-full blur-[120px]"></div>
                <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808008_1px,transparent_1px),linear-gradient(to_bottom,#80808008_1px,transparent_1px)] bg-[size:40px_40px]"></div>
            </div>

            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="w-full max-w-md relative z-10"
            >
                <Link href="/" className="inline-flex items-center gap-2 text-slate-500 hover:text-white transition-colors text-xs font-black uppercase tracking-widest mb-10 group">
                    <ChevronLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
                    Back to Terminal
                </Link>

                <div className="bg-[#111318] border border-white/5 p-10 rounded-[3rem] shadow-2xl backdrop-blur-3xl">
                    <div className="mb-10 flex flex-col items-center">
                        <div className="w-16 h-16 bg-orange-500 rounded-2xl flex items-center justify-center mb-6 shadow-xl shadow-orange-500/20">
                            <Shield className="w-8 h-8 text-black fill-current" />
                        </div>
                        <h1 className="text-3xl font-black text-white tracking-tighter">
                            {isLogin ? 'SECURE AUTHORIZATION' : 'REQUEST PROVISIONING'}
                        </h1>
                        <p className="text-slate-500 text-xs font-bold mt-2 uppercase tracking-widest italic opacity-50">
                            AES-256 Encrypted Tunnel v2.1
                        </p>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div className="space-y-2">
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest ml-1">Identity Endpoint (Email)</label>
                            <input
                                type="email" value={email} onChange={(e) => setEmail(e.target.value)} required
                                className="w-full bg-black/40 border border-white/5 rounded-2xl px-6 py-4 text-white focus:outline-none focus:border-orange-500/40 transition-all font-bold text-sm"
                                placeholder="name@hq.company"
                            />
                        </div>
                        <div className="space-y-2">
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest ml-1">Security Key (Password)</label>
                            <input
                                type="password" value={password} onChange={(e) => setPassword(e.target.value)} required
                                className="w-full bg-black/40 border border-white/5 rounded-2xl px-6 py-4 text-white focus:outline-none focus:border-orange-500/40 transition-all font-bold text-sm"
                                placeholder="••••••••"
                            />
                        </div>

                        {error && (
                            <div className="bg-red-500/10 border border-red-500/20 p-4 rounded-2xl text-red-500 text-[10px] font-black uppercase tracking-widest mt-4">
                                ⚠ {error}
                            </div>
                        )}

                        <button
                            disabled={loading}
                            className="w-full bg-white text-black font-black py-5 rounded-2xl mt-6 uppercase tracking-[0.2em] text-xs hover:bg-orange-500 hover:text-white transition-all transform active:scale-[0.98] disabled:opacity-50"
                        >
                            {loading ? 'PROCESSING...' : isLogin ? 'INITIATE ACCESS' : 'CREATE IDENTITY'}
                        </button>

                        <div className="relative py-6">
                            <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-white/5"></div></div>
                            <div className="relative flex justify-center text-[9px] font-black uppercase tracking-widest"><span className="bg-[#111318] px-4 text-slate-600">Enterprise SSO</span></div>
                        </div>

                        <button
                            type="button"
                            onClick={handleGoogleLogin}
                            className="w-full bg-white/[0.03] border border-white/10 text-white font-bold py-4 rounded-2xl flex items-center justify-center gap-3 hover:bg-white/[0.08] transition-all transform active:scale-[0.98]"
                        >
                            <svg className="w-4 h-4" viewBox="0 0 24 24">
                                <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                                <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                                <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" />
                                <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
                            </svg>
                            <span className="text-[10px] uppercase font-black tracking-widest">Verify with Workspace</span>
                        </button>
                    </form>

                    <div className="mt-10 text-center">
                        <button
                            onClick={() => setIsLogin(!isLogin)}
                            className="text-[10px] font-black text-slate-500 hover:text-orange-500 transition-all uppercase tracking-[0.2em]"
                        >
                            {isLogin ? "Request Enterprise Seat" : "Internal Login Protocol"}
                        </button>
                    </div>
                </div>

                <div className="mt-8 flex justify-center gap-8">
                    <Link href="#" className="flex items-center gap-2 text-slate-600 hover:text-white transition-all text-[9px] font-black uppercase tracking-widest"><Github className="w-3 h-3" /> GitHub</Link>
                    <Link href="#" className="text-slate-600 hover:text-white transition-all text-[9px] font-black uppercase tracking-widest">Privacy Shield</Link>
                    <Link href="#" className="text-slate-600 hover:text-white transition-all text-[9px] font-black uppercase tracking-widest">Protocol status</Link>
                </div>
            </motion.div>
        </div>
    );
}

export default function LoginPage() {
    return (
        <Suspense fallback={<div>Loading authorization state...</div>}>
            <LoginContent />
        </Suspense>
    );
}
