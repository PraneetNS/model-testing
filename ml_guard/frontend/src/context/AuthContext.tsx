"use client";

import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';

interface AuthContextType {
    user: any | null;
    token: string | null;
    loading: boolean;
    logout: () => Promise<void>;
    signInWithGoogle: () => Promise<void>;
    isDev: boolean;
}

const AuthContext = createContext<AuthContextType>({
    user: null,
    token: null,
    loading: true,
    logout: async () => { },
    signInWithGoogle: async () => { },
    isDev: false,
});

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
    const [user, setUser] = useState<any | null>(null);
    const [token, setToken] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);
    const [isDev, setIsDev] = useState(false);
    const router = useRouter();
    const pathname = usePathname();
    // Keep a ref so the onAuthStateChanged callback always reads the latest pathname
    // without needing to be in the effect dependency array.
    const pathnameRef = useRef(pathname);

    // Sync the ref every render so callbacks always see the latest value.
    useEffect(() => { pathnameRef.current = pathname; }, [pathname]);

    useEffect(() => {
        const session_id = `sess_${Math.random().toString(36).substring(2, 10)}`;
        let unsubscribe: (() => void) | undefined;
        let isInitialLoad = true;

        const initAuth = async () => {
            try {
                const searchParams = new URLSearchParams(window.location.search);
                const forceBypass = searchParams.get('bypass') === 'true';

                const hasFirebaseConfig = !!process.env.NEXT_PUBLIC_FIREBASE_API_KEY;

                // If we are in dev and have NO Firebase keys OR if ?bypass=true is specified
                if (process.env.NODE_ENV === 'development' && (!hasFirebaseConfig || forceBypass)) {
                    console.info('ML Guard: Using Developer Identity (Bypass: ' + forceBypass + ')');
                    
                    // Set session in cookie for proxy
                    await fetch('/api/auth/session', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            token: 'dev-token-999',
                            api_key: process.env.NEXT_PUBLIC_API_KEY || 'dev-secret-key'
                        })
                    }).catch(err => console.error("Failed to set dev session:", err));

                    setIsDev(true);
                    setUser({
                        email: 'dev@local',
                        displayName: 'Developer Node',
                        uid: 'dev-001',
                        sessionId: session_id,
                        role: 'administrator'
                    });
                    setToken('dev-token-999');
                    setLoading(false);
                    return;
                }

                const { onAuthStateChanged } = await import('firebase/auth');
                const { auth } = await import('@/lib/firebase');

                unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
                    try {
                        if (currentUser) {
                            // Attempt to get token without forcing refresh first
                            const idToken = await currentUser.getIdToken().catch(err => {
                                if (err?.code === 'auth/network-request-failed') {
                                    throw err; // Re-throw to be caught by outer try-catch
                                }
                                return currentUser.getIdToken(true); // Retry once with force refresh if it was just a local issue
                            });
                            
                            // Set session in cookie
                            await fetch('/api/auth/session', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ 
                                    token: idToken,
                                    api_key: process.env.NEXT_PUBLIC_API_KEY
                                })
                            });

                            setUser({
                                uid: currentUser.uid,
                                email: currentUser.email,
                                displayName: currentUser.displayName || currentUser.email?.split('@')[0] || 'Operator',
                                photoURL: currentUser.photoURL,
                                providerData: currentUser.providerData,
                                sessionId: session_id
                            });
                            setToken(idToken);
                        } else {
                            setUser(null);
                            setToken(null);
                            fetch('/api/auth/session', { method: 'DELETE' }).catch(() => {});
                            if (pathnameRef.current.startsWith('/dashboard')) router.push('/login');
                        }
                    } catch (err: any) {
                        const isNetworkError = err?.code === 'auth/network-request-failed';
                        
                        if (process.env.NODE_ENV === 'development') {
                            if (isNetworkError) {
                                console.info('ML Guard: Firebase network unavailable, using Offline Developer identity.');
                            } else {
                                console.warn('Auth Warning:', err.message || err);
                            }
                            
                            setIsDev(true);
                            setUser({
                                email: 'dev@local',
                                displayName: 'Offline Developer',
                                uid: 'dev-001',
                                sessionId: session_id,
                                role: 'administrator'
                            });
                            setToken('dev-token-999');
                        } else {
                            if (!isNetworkError) console.error('Auth Error:', err);
                            setUser(null);
                            setToken(null);
                        }
                    }
                    setLoading(false);
                });
            } catch (e) {
                setIsDev(true);
                setUser({
                    email: 'dev@local',
                    displayName: 'Bypass Identity',
                    uid: 'bypass-001',
                    sessionId: session_id
                });
                setToken('dev-token');
                setLoading(false);
            }
        };

        if (typeof window !== 'undefined') {
            initAuth();
        }

        return () => {
            if (unsubscribe) unsubscribe();
        };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); // Run once on mount — pathnameRef keeps the latest pathname in callbacks

    const logout = async () => {
        try {
            const { signOut } = await import('firebase/auth');
            const { auth } = await import('@/lib/firebase');
            await signOut(auth);
            await fetch('/api/auth/session', { method: 'DELETE' });
        } catch { }

        setUser(null);
        setToken(null);
        setIsDev(false);
        router.push('/');
    };

    const signInWithGoogle = async () => {
        console.log('Google Sign-In started...');
        try {
            if (!process.env.NEXT_PUBLIC_FIREBASE_API_KEY) {
                console.error('Firebase API Key is missing! Check your .env file.');
                alert('Authentication configuration is missing. Please contact support.');
                return;
            }

            const { GoogleAuthProvider, signInWithPopup } = await import('firebase/auth');
            const { auth } = await import('@/lib/firebase');
            const provider = new GoogleAuthProvider();
            
            console.log('Opening Google Sign-In popup...');
            const result = await signInWithPopup(auth, provider);
            console.log('Google Sign-In successful:', result.user.email);
            
            router.push('/dashboard');
        } catch (error: any) {
            console.error('Google Sign-In Error:', error);
            if (error.code === 'auth/popup-blocked') {
                alert('Sign-in popup was blocked by your browser. Please allow popups for this site.');
            } else {
                alert(`Sign-in failed: ${error.message}`);
            }
        }
    };

    return (
        <AuthContext.Provider value={{ user, token, loading, logout, signInWithGoogle, isDev }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);
