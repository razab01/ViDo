import React from 'react';
import { Eye, History, MessageSquareOff, Shield } from 'lucide-react';

export const LandingBenefits: React.FC = () => {
  const principles = [
    {
      icon: <Eye className="text-[var(--accent)]" aria-hidden="true" />,
      title: 'Clarity over complexity',
      desc: 'No endless configuration screens or micro-toggles. Everything is organized so you can answer what needs to be done in seconds.',
    },
    {
      icon: <History className="text-[var(--green)]" aria-hidden="true" />,
      title: 'History over guesswork',
      desc: 'Measurable progress checkpoints and completion timestamps replace vague "in progress" statuses with observable evidence.',
    },
    {
      icon: <Shield className="text-[var(--amber)]" aria-hidden="true" />,
      title: 'Privacy by default',
      desc: 'Your private tasks are strictly your own. Sharing is an intentional action, keeping your personal life separate from work.',
    },
    {
      icon: <MessageSquareOff className="text-indigo-600" aria-hidden="true" />,
      title: 'Collaboration without noise',
      desc: 'Clear task ownership and chronological checkpoint timelines answer "what is the status?" without scheduling another meeting.',
    },
  ];

  return (
    <section id="benefits" className="vdb-section vdb-benefits">
      <div className="vdb-section-heading">
        <span className="vdb-section-label">Product Principles</span>
        <h2 className="vdb-section-title">Built on principles that respect your attention</h2>
        <p className="vdb-section-copy">
          ViDo is purposefully restrained. We resist adding features that clutter your day with unnecessary notifications or guilt.
        </p>
      </div>

      <div className="vdb-principle-grid">
        {principles.map((p, i) => (
          <article key={p.title} className="vdb-principle-card">
            <span className="vdb-principle-index">
              {String(i + 1).padStart(2, '0')} / 04
            </span>
            <span className="vdb-principle-icon">{p.icon}</span>
            <h3 className="vdb-principle-title">{p.title}</h3>
            <p className="vdb-principle-desc">{p.desc}</p>
          </article>
        ))}
      </div>
    </section>
  );
};
