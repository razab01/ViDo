import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertCircle, X } from 'lucide-react';
import { LandingHeader } from './LandingHeader';
import { LandingHero } from './LandingHero';
import { LandingPreview } from './LandingPreview';
import { LandingUseCases } from './LandingUseCases';
import { LandingBenefits } from './LandingBenefits';
import { LandingFooter } from './LandingFooter';
import { NoticeModal } from '../../components/ui';
import { useAuth } from '../auth';
import './landing.css';

export const LandingPage: React.FC = () => {
  const [isNoticeOpen, setIsNoticeOpen] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);
  const { user, signInWithGoogle } = useAuth();
  const navigate = useNavigate();

  const handleSignIn = async () => {
    setIsNoticeOpen(false);
    setAuthError(null);
    if (user) {
      navigate('/workspace');
      return;
    }

    try {
      await signInWithGoogle();
      navigate('/workspace');
    } catch (err: any) {
      console.error('Sign-in failed:', err);
      const isNetworkOrOffline =
        err?.code === 'auth/network-request-failed' ||
        err?.code === 'auth/operation-not-supported-in-this-environment' ||
        err?.message?.includes('fetch') ||
        err?.message?.includes('network') ||
        err?.message?.includes('failed to fetch');

      if (isNetworkOrOffline) {
        setAuthError(
          'Firebase Auth Emulator is offline on localhost:9099. Please start the emulator with "npm run dev" or "npm run emulators" to sign in.'
        );
      } else if (err?.code === 'auth/popup-closed-by-user') {
        setAuthError('Sign-in was cancelled (popup was closed).');
      } else {
        setAuthError(
          err?.message || 'Authentication failed. Please verify that the Firebase Auth Emulator is active.'
        );
      }
    }
  };

  return (
    <div className="vdb-page flex min-h-screen flex-col justify-between">
      <div>
        <LandingHeader onGoogleSignInClick={() => setIsNoticeOpen(true)} />

        {authError && (
          <div role="alert" className="vdb-auth-error">
            <div className="flex items-start gap-2.5">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{authError}</span>
            </div>
            <button
              type="button"
              onClick={() => setAuthError(null)}
              aria-label="Dismiss error"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        <main>
          <LandingHero onGoogleSignInClick={() => setIsNoticeOpen(true)} />
          <LandingPreview />
          <LandingUseCases />
          <LandingBenefits />
        </main>
      </div>

      <LandingFooter />

      <NoticeModal
        isOpen={isNoticeOpen}
        onClose={() => setIsNoticeOpen(false)}
        title="Welcome to ViDo"
        description="You can enter and explore your workspace directly right now without needing to sign in."
        onSignIn={handleSignIn}
      />
    </div>
  );
};
