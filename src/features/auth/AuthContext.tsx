import React, { createContext, useContext, useEffect, useState, useMemo } from 'react';
import {
  signInWithPopup,
  signOut as firebaseSignOut,
  onAuthStateChanged,
} from 'firebase/auth';
import { getFirebaseInstance, type AuthMode } from './firebase';

export interface AuthUser {
  uid: string;
  displayName: string | null;
  email: string | null;
  photoURL?: string | null;
}

export interface AuthContextValue {
  user: AuthUser | null;
  loading: boolean;
  authMode: AuthMode;
  signInWithGoogle: () => Promise<void>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  const { auth, googleProvider, authMode } = useMemo(() => getFirebaseInstance(), []);

  useEffect(() => {
    if (!auth) {
      setLoading(false);
      return;
    }

    const unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
      if (firebaseUser) {
        setUser({
          uid: firebaseUser.uid,
          displayName: firebaseUser.displayName,
          email: firebaseUser.email,
          photoURL: firebaseUser.photoURL,
        });
      } else {
        setUser(null);
      }
      setLoading(false);
    });

    return () => unsubscribe();
  }, [auth]);

  const signInWithGoogle = async () => {
    if (!auth || !googleProvider) {
      console.warn('Firebase Auth is not initialized.');
      return;
    }
    try {
      const result = await signInWithPopup(auth, googleProvider);
      // Publish the signed-in user before this promise resolves. Otherwise a
      // caller that navigates immediately can reach a guard that still sees no
      // user and bounce back to the landing page.
      const firebaseUser = result?.user;
      if (firebaseUser) {
        setUser({
          uid: firebaseUser.uid,
          displayName: firebaseUser.displayName,
          email: firebaseUser.email,
          photoURL: firebaseUser.photoURL,
        });
      }
    } catch (error) {
      console.error('Google Sign-In error:', error);
      throw error;
    }
  };

  const signOut = async () => {
    if (auth) {
      await firebaseSignOut(auth);
    }
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        authMode,
        signInWithGoogle,
        signOut,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

const defaultAuthContext: AuthContextValue = {
  user: null,
  loading: false,
  authMode: 'emulator',
  signInWithGoogle: async () => {},
  signOut: async () => {},
};

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  return context || defaultAuthContext;
}
