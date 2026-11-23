import React, { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Search, Sparkles } from 'lucide-react';
import { Badge, Button, Card } from '../../components/ui';
import { profileData, ragDocs, runRAG, USER_RAG_MODE } from './engine';
import type { RAGResult } from './engine';
import { readSnapshot, subscribeSnapshot } from './mlStore';
import { polishRagAnswer } from './answerEngine';
import { RagWorking } from './WorkingIndicator';

export interface DashboardRagCardProps {
  userId?: string;
  basePath?: string;
}

export const DashboardRagCard: React.FC<DashboardRagCardProps> = ({
  userId = 'demo-user-uid',
  basePath = '/preview',
}) => {
  const [snapshot, setSnapshot] = useState(() => readSnapshot(userId));
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState<RAGResult | null>(null);
  const [notice, setNotice] = useState('');
  const [ragBusy, setRagBusy] = useState(false);
  const [ragStage, setRagStage] = useState('');

  useEffect(() => {
    setSnapshot(readSnapshot(userId));
    return subscribeSnapshot(userId, () => setSnapshot(readSnapshot(userId)));
  }, [userId]);

  useEffect(() => {
    setAnswer(null);
    setNotice('');
    setRagBusy(false);
    setRagStage('');
  }, [userId]);

  const profile = useMemo(() => snapshot?.profile ?? profileData([], []), [snapshot]);
  const suggestions = snapshot?.result?.suggestions ?? [];
  const docs = useMemo(
    () => ragDocs(snapshot?.dataset ?? null, profile, suggestions),
    [snapshot, profile, suggestions]
  );

  const handleAsk = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!docs.length || !query.trim()) {
      setNotice('Type a question about your saved data.');
      return;
    }
    const question = query.trim();
    setNotice('');
    setRagBusy(true);
    setRagStage('');
    setAnswer(null);
    const startedAt = Date.now();
    await new Promise<void>((resolve) => window.setTimeout(resolve, 30));
    try {
      const retrieved = runRAG(question, USER_RAG_MODE, docs, profile, suggestions);
      setRagStage('Comparing the best matches…');
      const next = await polishRagAnswer({ question, rag: retrieved, onProgress: setRagStage });
      const remaining = 420 - (Date.now() - startedAt);
      if (remaining > 0) {
        await new Promise<void>((resolve) => window.setTimeout(resolve, remaining));
      }
      setAnswer(next);
    } finally {
      setRagBusy(false);
      setRagStage('');
    }
  };

  const answerLines = useMemo(() => {
    if (!answer?.answer) return [];
    const parts = answer.answer.split(/\.\s+/).map((part) => part.trim()).filter(Boolean);
    return parts.map((part, index) => (index < parts.length - 1 ? `${part}.` : part));
  }, [answer]);

  return (
    <Card className="border-[var(--accent)]/20">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Badge variant="accent" size="sm">New</Badge>
            <h2 className="text-sm font-bold flex items-center gap-2">
              <Search className="w-4 h-4 text-[var(--accent)]" /> Ask your data
            </h2>
          </div>
          <p className="text-xs text-[var(--muted)] max-w-xl">
            ViDo searches the spreadsheet you saved in ML Lab and writes a clear answer for you.
            {snapshot?.upload && (
              <span className="block text-[11px] text-[var(--faint)] mt-0.5">
                Auto-detected source: {snapshot.upload.name}
                {snapshot.sheetName ? ` · ${snapshot.sheetName}` : ''}
              </span>
            )}
          </p>
        </div>
        <Link
          to={`${basePath}/ml-lab`}
          className="inline-flex items-center gap-1 text-xs font-semibold text-[var(--accent)] hover:underline whitespace-nowrap"
        >
          Open ML Lab <ArrowRight className="w-3.5 h-3.5" />
        </Link>
      </div>

      <div className="flex flex-wrap items-center gap-2 mt-3 text-[11px] text-[var(--muted)]">
        {snapshot?.result ? (
          <>
            <Badge size="sm" variant="success">Model ready</Badge>
            <span>{docs.length} searchable sources</span>
          </>
        ) : (
          <span>No saved dataset yet. Upload a CSV or Excel file in ML Lab to enable RAG here.</span>
        )}
      </div>

      {docs.length > 0 && (
        <form onSubmit={handleAsk} className="flex flex-col sm:flex-row gap-2 mt-3">
          <input
            aria-label="Ask your data"
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setNotice('');
            }}
            placeholder="Ask a question about your saved rows…"
            className="flex-1 px-3 py-2 text-sm bg-[var(--surface)] border border-[var(--line)] rounded-lg"
          />
          <Button type="submit" variant="secondary" size="sm" disabled={ragBusy}>
            {ragBusy ? 'Finding…' : 'Find answer'}
          </Button>
        </form>
      )}
      {notice && <p className="text-[11px] text-[var(--muted)] mt-2">{notice}</p>}

      {ragBusy && <RagWorking compact stage={ragStage} />}

      {answer && !ragBusy && (
        <div className="mt-3 rounded-lg border border-[var(--line)] bg-[var(--surface)] p-3">
          <div className="flex items-center gap-2 mb-2">
            <Sparkles className="w-3.5 h-3.5 text-[var(--accent)]" />
            <span className="text-[11px] font-bold uppercase tracking-wide text-[var(--muted)]">
              Answer
            </span>
          </div>
          {answerLines.length ? (
            <ul className="space-y-1.5 text-xs text-[var(--ink)]">
              {answerLines.map((line, index) => (
                <li key={`${line}-${index}`} className="flex gap-2">
                  <span className="text-[var(--accent)] mt-1.5 text-[8px]">●</span>
                  <span>{line}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-[var(--muted)]">{answer.answer}</p>
          )}
        </div>
      )}
    </Card>
  );
};

export default DashboardRagCard;
