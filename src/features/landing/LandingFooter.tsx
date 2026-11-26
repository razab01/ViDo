import React from 'react';
import { Link } from 'react-router-dom';

export const LandingFooter: React.FC = () => {
  return (
    <footer className="vdb-footer">
      <div className="vdb-footer-inner">
        <div>
          <div className="vdb-footer-brand">
            <span>V</span>
            <span>ViDo</span>
          </div>
          <p className="vdb-footer-tagline">
            Simple enough for anyone. Structured enough for serious work.
          </p>
        </div>

        <nav className="vdb-footer-links" aria-label="Footer navigation">
          <Link to="/preview">Live Preview</Link>
          <Link to="/style-lab">Themes</Link>
          <a
            href="https://github.com/vicharanashala/ViDo"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub Repository
          </a>
        </nav>
      </div>

      <div className="vdb-footer-bottom">&copy; 2026 ViDo. All rights reserved.</div>
    </footer>
  );
};
