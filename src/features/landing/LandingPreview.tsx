import React from 'react';
import { AlertCircle, CheckCircle2, Clock } from 'lucide-react';

export const LandingPreview: React.FC = () => {
  return (
    <section id="preview" className="vdb-section">
      <div className="vdb-section-heading">
        <span className="vdb-section-label">Dashboard Overview</span>
        <h2 className="vdb-section-title">Designed for clarity. Built without clutter.</h2>
        <p className="vdb-section-copy">
          See how ViDo organizes your responsibilities into calm, section-limited views that keep you focused on what matters next.
        </p>
      </div>

      <div className="vdb-showcase">
        <div className="vdb-showcase-inner">
          <div className="vdb-showcase-chrome">
            <div className="vdb-chrome-left">
              <span className="vdb-chrome-dot" style={{ background: '#e78989' }} />
              <span className="vdb-chrome-dot" style={{ background: '#e6bb6a' }} />
              <span className="vdb-chrome-dot" style={{ background: '#77b596' }} />
              <span className="vdb-chrome-url">vido.app/preview</span>
            </div>
            <span className="vdb-showcase-date">Today: 17 Nov 2026</span>
          </div>

          <div className="vdb-showcase-body">
            <aside className="vdb-mock-sidebar">
              <div className="vdb-mock-brand">
                <span>V</span>
                ViDo
              </div>
              <div className="vdb-mock-nav">
                <div className="vdb-mock-nav-item active">
                  <span>Home</span>
                  <span>11</span>
                </div>
                <div className="vdb-mock-nav-item">
                  <span>Today</span>
                  <span>4</span>
                </div>
                <div className="vdb-mock-nav-item">
                  <span>Upcoming</span>
                  <span>5</span>
                </div>
                <div className="vdb-mock-nav-item">
                  <span>Needs Attention</span>
                  <span>2</span>
                </div>
                <div className="vdb-mock-nav-item">
                  <span>Completed</span>
                  <span>8</span>
                </div>
              </div>
            </aside>

            <div className="vdb-mock-main">
              <div className="vdb-mock-topbar">
                <span>Search tasks or filter by priority...</span>
                <span className="vdb-mock-avatar">DU</span>
              </div>

              <div className="vdb-mock-grid">
                <div className="vdb-mock-column">
                  <div className="vdb-mock-heading">
                    <h3>
                      <Clock aria-hidden="true" /> Today (4)
                    </h3>
                    <span className="vdb-mock-link">View all</span>
                  </div>
                  <div className="vdb-mock-task">
                    <div className="vdb-mock-task-title">
                      <span>Review Q4 Budget &amp; Resource Allocations</span>
                      <span className="vdb-mock-pill blue">High</span>
                    </div>
                    <span className="vdb-mock-task-meta">Due Today at 17:00</span>
                  </div>
                  <div className="vdb-mock-task">
                    <div className="vdb-mock-task-title">
                      <span>Finalize Vehicle Registration Renewal</span>
                      <span className="vdb-mock-pill grey">Medium</span>
                    </div>
                    <span className="vdb-mock-task-meta">Due Today at 18:30</span>
                  </div>
                </div>

                <div className="vdb-mock-column">
                  <div className="vdb-mock-heading">
                    <h3>
                      <AlertCircle aria-hidden="true" /> Needs Attention (2)
                    </h3>
                    <span className="vdb-mock-link">View all</span>
                  </div>
                  <div className="vdb-mock-task alert">
                    <div className="vdb-mock-task-title">
                      <span>Submit Annual Insurance Declaration</span>
                      <span className="vdb-mock-pill red">Overdue</span>
                    </div>
                    <span className="vdb-mock-task-meta red">Was due 15 Nov 2026</span>
                  </div>
                  <div className="vdb-mock-task warn">
                    <div className="vdb-mock-task-title">
                      <span>Server Security Certificate Expiry</span>
                      <span className="vdb-mock-pill amber">Urgent</span>
                    </div>
                    <span className="vdb-mock-task-meta amber">Expires in 24 hours</span>
                  </div>
                </div>
              </div>

              <div className="vdb-mock-completed">
                <div className="vdb-mock-heading">
                  <h3>
                    <CheckCircle2 aria-hidden="true" /> Recently Completed (Last 7 Days)
                  </h3>
                  <span className="vdb-mock-link">View all</span>
                </div>
                <div className="vdb-mock-completed-row">
                  <span>Complete initial workspace design review</span>
                  <span>Completed on time</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
