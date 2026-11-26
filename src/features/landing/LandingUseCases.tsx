import React from 'react';
import { Briefcase, CheckSquare, Home, User } from 'lucide-react';

export const LandingUseCases: React.FC = () => {
  return (
    <section id="use-cases" className="vdb-section">
      <div className="vdb-section-heading">
        <span className="vdb-section-label">Universal Application</span>
        <h2 className="vdb-section-title">Built for every area of life</h2>
        <p className="vdb-section-copy">
          From personal doctor appointments to shared family chores and collaborative team releases, ViDo adapts without requiring separate apps.
        </p>
      </div>

      <div className="vdb-usecase-grid">
        <article className="vdb-usecase-card flex flex-col justify-between">
          <span className="vdb-usecase-no">01</span>
          <div>
            <span className="vdb-usecase-icon">
              <User aria-hidden="true" />
            </span>
            <h3 className="vdb-usecase-title">Personal Tasks</h3>
            <p className="vdb-usecase-desc">
              A private space for individual responsibilities, daily errands, study schedules, and self-directed projects.
            </p>
            <ul className="vdb-usecase-list">
              <li>
                <CheckSquare aria-hidden="true" />
                Precise target dates and deadline times
              </li>
              <li>
                <CheckSquare aria-hidden="true" />
                Clear active, overdue, and completed separation
              </li>
              <li>
                <CheckSquare aria-hidden="true" />
                Private by default with strict user isolation
              </li>
            </ul>
          </div>
          <span className="vdb-usecase-example">
            Example: Renew driver's license, Finish Chapter 4
          </span>
        </article>

        <article className="vdb-usecase-card vdb-usecase-card--home flex flex-col justify-between">
          <span className="vdb-usecase-no">02</span>
          <div>
            <span className="vdb-usecase-icon">
              <Home aria-hidden="true" />
            </span>
            <h3 className="vdb-usecase-title">Household &amp; Living</h3>
            <p className="vdb-usecase-desc">
              Keep household maintenance, utility deadlines, property taxes, and family chores transparent and on track.
            </p>
            <ul className="vdb-usecase-list">
              <li>
                <CheckSquare aria-hidden="true" />
                Prevent missed utility bills and service dates
              </li>
              <li>
                <CheckSquare aria-hidden="true" />
                Section limits keep lists approachable
              </li>
              <li>
                <CheckSquare aria-hidden="true" />
                Honest status without guilt or stress
              </li>
            </ul>
          </div>
          <span className="vdb-usecase-example">
            Example: HVAC winter maintenance, Electricity bill
          </span>
        </article>

        <article className="vdb-usecase-card vdb-usecase-card--work flex flex-col justify-between">
          <span className="vdb-usecase-no">03</span>
          <div>
            <span className="vdb-usecase-icon">
              <Briefcase aria-hidden="true" />
            </span>
            <h3 className="vdb-usecase-title">Work &amp; Projects</h3>
            <p className="vdb-usecase-desc">
              Structured accountability for deliverables, sprint targets, client roadmaps, and cross-functional commitments.
            </p>
            <ul className="vdb-usecase-list">
              <li>
                <CheckSquare aria-hidden="true" />
                Explicit owner, priority, and due dates
              </li>
              <li>
                <CheckSquare aria-hidden="true" />
                Chronological update and checkpoint history
              </li>
              <li>
                <CheckSquare aria-hidden="true" />
                Designed for complete team visibility and transparent execution
              </li>
            </ul>
          </div>
          <span className="vdb-usecase-example">
            Example: Prepare Q4 budget, Deploy v1.2 release
          </span>
        </article>
      </div>
    </section>
  );
};
