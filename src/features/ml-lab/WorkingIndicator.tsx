import React, { useEffect, useState } from 'react';
import { CheckCircle2, Loader2, Search, Sparkles } from 'lucide-react';

const TRAINING_STEPS = [
  'Reading your file',
  'Finding useful patterns',
  'Choosing the best approach',
  'Almost ready',
];

const RAG_STEPS = [
  'Understanding your question',
  'Searching your saved data',
  'Comparing the best matches',
  'Putting the answer together',
];

export interface TrainingProgressProps {
  progress: number;
  detail?: string;
}

export const TrainingProgress: React.FC<TrainingProgressProps> = ({ progress, detail }) => {
  const [step, setStep] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(() => setStep((current) => (current + 1) % TRAINING_STEPS.length), 700);
    return () => window.clearInterval(timer);
  }, []);

  const safeProgress = Math.min(100, Math.max(0, Math.round(progress)));
  const label = safeProgress >= 90 ? TRAINING_STEPS[3] : TRAINING_STEPS[step % 3];

  return (
    <div
      role="status"
      aria-live="polite"
      className="rounded-2xl border border-[var(--accent)]/25 bg-[var(--surface)] p-5 shadow-sm"
    >
      <div className="flex items-center gap-3.5">
        <div className="relative flex h-11 w-11 shrink-0 items-center justify-center">
          <span className="absolute inset-0 rounded-full bg-[var(--accent-soft)] animate-ping opacity-70" />
          <span className="relative flex h-10 w-10 items-center justify-center rounded-full bg-[var(--accent)] text-white shadow-lg">
            <Loader2 className="h-5 w-5 animate-spin" />
          </span>
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <Sparkles className="h-3.5 w-3.5 shrink-0 text-[var(--accent)]" />
            <p className="text-sm font-bold text-[var(--ink)]">Learning from your file…</p>
          </div>
          <p className="mt-0.5 truncate text-[11px] text-[var(--muted)]">{label}</p>
        </div>
        <span className="rounded-full bg-[var(--canvas)] px-2.5 py-1 text-[11px] font-bold text-[var(--accent)]">
          {safeProgress}%
        </span>
      </div>
      <div className="mt-4 h-2 overflow-hidden rounded-full bg-[var(--canvas)]">
        <div
          className="h-full rounded-full bg-[var(--accent)] transition-all duration-500"
          style={{ width: `${safeProgress}%` }}
        />
      </div>
      {detail && <p className="mt-2 truncate text-[10px] text-[var(--faint)]">{detail}</p>}
    </div>
  );
};

export interface RagWorkingProps {
  compact?: boolean;
}

export const RagWorking: React.FC<RagWorkingProps> = ({ compact = false }) => {
  const [step, setStep] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(() => setStep((current) => (current + 1) % RAG_STEPS.length), 520);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <div
      role="status"
      aria-live="polite"
      className={`${compact ? 'p-3' : 'p-4'} mt-3 rounded-xl border border-[var(--accent)]/25 bg-[var(--accent-soft)]/60`}
    >
      <div className="flex items-center gap-3">
        <div className="relative flex h-9 w-9 shrink-0 items-center justify-center">
          <span className="absolute inset-0 rounded-full bg-[var(--accent)]/20 animate-ping" />
          <span className="relative flex h-8 w-8 items-center justify-center rounded-full bg-[var(--surface)] text-[var(--accent)] shadow-sm">
            <Search className="h-4 w-4" />
          </span>
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <Loader2 className="h-3.5 w-3.5 animate-spin text-[var(--accent)]" />
            <p className="text-xs font-bold text-[var(--ink)]">Finding the answer…</p>
          </div>
          <p className="mt-0.5 truncate text-[11px] text-[var(--muted)]">{RAG_STEPS[step]}</p>
        </div>
        <span className="flex items-end gap-1" aria-hidden="true">
          {[0, 1, 2].map((index) => (
            <span
              key={index}
              className="h-1.5 w-1.5 rounded-full bg-[var(--accent)] animate-pulse"
              style={{ animationDelay: `${index * 140}ms` }}
            />
          ))}
        </span>
      </div>
      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-[var(--surface)]/80">
        <div className="h-full w-2/3 animate-pulse rounded-full bg-[var(--accent)]" />
      </div>
    </div>
  );
};

export interface RAGDoneProps {
  compact?: boolean;
}

export const RAGDone: React.FC<RAGDoneProps> = ({ compact = false }) => (
  <div
    className={`${compact ? 'px-3 py-2 text-[10px]' : 'px-4 py-2.5 text-[11px]'} flex items-center gap-2 rounded-xl border border-[var(--green)]/25 bg-[var(--green-soft)] text-[var(--green)]`}
  >
    <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
    <span className="font-semibold">Answer ready</span>
  </div>
);
