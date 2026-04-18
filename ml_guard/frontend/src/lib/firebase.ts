import { initializeApp, getApps, getApp } from "firebase/app";
import { Auth, getAuth, initializeAuth, browserLocalPersistence } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
    apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
    authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
    projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
    storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
    messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
    appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID
};

// Initialize Firebase
const app = getApps().length > 0 ? getApp() : initializeApp(firebaseConfig);

// Initialize Firebase Auth with persistence explicitly to avoid internal race conditions in Next.js
let auth: Auth;
if (getApps().length > 0) {
    auth = getAuth(app);
} else {
    try {
        auth = initializeAuth(app, {
            persistence: browserLocalPersistence
        });
    } catch (e) {
        // already initialized
        auth = getAuth(app);
    }
}

const db = getFirestore(app);

export { app, auth, db };
