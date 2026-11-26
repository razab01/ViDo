import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Sparkles } from 'lucide-react';
import { useAuth } from '../auth';

export interface LandingHeroProps {
  onGoogleSignInClick?: () => void;
}

export const LandingHero: React.FC<LandingHeroProps> = ({ onGoogleSignInClick }) => {
  const { user, signInWithGoogle } = useAuth();

  const handleSignIn = () => {
    if (onGoogleSignInClick) {
      onGoogleSignInClick();
    } else {
      void signInWithGoogle();
    }
  };

  return (
    <section className="vdb-hero">
      <div className="vdb-hero-grid">
        <div className="vdb-hero-copy">
          <div className="vdb-eyebrow">
            <Sparkles aria-hidden="true" />
            A calmer way to get things done
          </div>

          <h1 className="vdb-hero-title">
            Simple enough for anyone.{' '}
            <span className="vdb-hero-accent">Structured enough for serious work.</span>
          </h1>

          <p className="vdb-hero-lede">
            A calm, focused task workspace designed for personal errands, household responsibilities, and professional deadlines—without turning every status update into a meeting.
          </p>

          <div className="vdb-hero-actions">
            {user ? (
              <Link to="/workspace" className="vdb-cta-primary">
                <span>Go to Workspace</span>
                <ArrowRight aria-hidden="true" />
              </Link>
            ) : (
              <button type="button" onClick={handleSignIn} className="vdb-cta-primary">
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path
                    fill="#FFFFFF"
                    d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                  />
                  <path
                    fill="#FFFFFF"
                    d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                  />
                  <path
                    fill="#FFFFFF"
                    d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
                  />
                  <path
                    fill="#FFFFFF"
                    d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
                  />
                </svg>
                <span>Continue with Google</span>
              </button>
            )}

            <Link to="/preview" className="vdb-cta-secondary">
              <span>Live Preview</span>
              <ArrowRight aria-hidden="true" />
            </Link>
          </div>

          <p className="vdb-hero-note">Free to use. Simple, focused, and distraction-free.</p>
        </div>

        <div className="vdb-hero-visual" aria-hidden="true">
          <div className="vdb-hero-orbit">
            <span className="vdb-orbit-dot vdb-orbit-dot--one" />
            <span className="vdb-orbit-dot vdb-orbit-dot--two" />
          </div>
          <div className="vdb-hero-core">
            <span>ViDo</span>
            <span>one calm workspace</span>
          </div>
          <div className="vdb-float-card vdb-float-card--top">
            <span className="vdb-float-label">Today</span>
            <strong className="vdb-float-value">4 priorities in focus</strong>
            <div className="vdb-float-meta">
              <i /> Nothing urgent yet
            </div>
          </div>
          <div className="vdb-float-card vdb-float-card--bottom">
            <span className="vdb-float-label">Domains</span>
            <strong className="vdb-float-value">3 areas in balance</strong>
            <div className="vdb-float-meta">
              <i /> Personal, home, and work
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
