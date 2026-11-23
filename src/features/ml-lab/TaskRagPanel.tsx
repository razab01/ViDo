import React, { useEffect, useMemo, useState } from 'react';
import { Search, Sparkles } from 'lucide-react';
import { Badge, Button } from '../../components/ui';
import type { SampleTask } from '../../data/sampleTasks';
import { profileData, ragDocs, runRAG, USER_RAG_MODE } from './engine';
import type { RAGDoc, RAGResult } from './engine';
import {
  formatTaskProbability,
  predictTask,
  readSnapshot,
  requestRestore,
  subscribeActiveResult,
  subscribeSnapshot,
} from './mlStore';
import type { TaskPrediction } from './mlStore';
import { polishRagAnswer } from './answerEngine';
import { KNOWLEDGE_DOCS } from './knowledge';
import { RagWorking } from './WorkingIndicator';

export interface TaskRagPanelProps {
  task: SampleTask;
  userId?: string;
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

const formatTaskDate = (value?: string): string => {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value || '');
  if (!match) return value || 'an unset date';
  const [, year, month, day] = match;
  return `${Number(day)} ${MONTHS[Number(month) - 1] || month} ${year}`;
};

const stripIdFragments = (text: string): string =>
  text
    .replace(/\b(?:task[_-]?id|source[_-]?row|row[_-]?id|index)\s*:\s*[A-Za-z0-9_-]+\b/gi, '')
    .replace(/\s{2,}/g, ' ')
    .trim();

/**
 * Questions about the selected task are answered from its structured fields so
 * saved spreadsheet rows cannot leak unrelated ids or dates into the answer.
 */
const directTaskAnswer = (
  query: string,
  task: SampleTask,
  prediction: TaskPrediction | null
): string | null => {
  const q = query.toLowerCase();
  const due = formatTaskDate(task.dueDate);
  const dueText = task.dueTime ? `${due} at ${task.dueTime}` : due;

  if (/\bwhen\b|\bdue\b|\bdeadline\b|by when/.test(q)) {
    return task.dueDate
      ? `Due: ${dueText}.`
      : 'This task does not have a due date yet.';
  }

  if (/\babout\b|\bdescribe\b|\bsummary\b|\bsummarize\b/.test(q)) {
    if (task.description) return `This task is about ${task.description.replace(/\.$/, '')}.`;
    return `This task is “${task.title}” in the ${task.category || 'general'} category.`;
  }

  if (/probab|likelihood|chance|\bscore\b/.test(q)) {
    if (!prediction) return 'No trained model probability is available for this task yet.';
    const label = prediction.isCompletion ? 'Completion probability' : 'Estimated outcome';
    const chance = formatTaskProbability(prediction.probability, prediction.kind);
    return prediction.isCompletion
      ? `${label}: ${chance}.`
      : `${label}: ${chance} ${prediction.label}.`;
  }

  if (/what should i do next|\bnext step\b|\bnext\b/.test(q)) {
    if (task.state === 'completed') return 'This task is already complete.';
    const latest = task.checkpoints?.length
      ? ` Latest update: ${task.checkpoints[task.checkpoints.length - 1].note}`
      : '';
    return `Next: continue “${task.title}”${task.priority ? ` (${task.priority} priority)` : ''}, due ${dueText}.${latest}`;
  }

  return null;
};

const taskFactLines = (task: SampleTask, prediction: TaskPrediction | null): string[] =>
  [
    `Task title: ${task.title}.`,
    task.description ? `Description: ${task.description}.` : '',
    `State: ${task.state}.`,
    task.priority ? `Priority: ${task.priority}.` : '',
    task.category ? `Category: ${task.category}.` : '',
    task.startDate ? `Start date: ${task.startDate}.` : '',
    task.dueDate ? `Due date: ${task.dueDate}.` : '',
    task.dueTime ? `Due time: ${task.dueTime}.` : '',
    task.createdAt ? `Created: ${task.createdAt}.` : '',
    prediction
      ? `${prediction.isCompletion ? 'Completion probability' : 'Estimated outcome'}: ${formatTaskProbability(
          prediction.probability,
          prediction.kind
        )}${prediction.isCompletion ? '' : ` ${prediction.label}`}.`
      : '',
    ...(task.checkpoints || []).map((checkpoint) => `Checkpoint: ${checkpoint.note}.`),
  ].filter(Boolean);

const taskToRagDoc = (task: SampleTask, prediction: TaskPrediction | null): RAGDoc => ({
  id: `task-${task.id}`,
  title: task.title,
  text: taskFactLines(task, prediction).join(' '),
  tags: `category=${task.category || ''}, priority=${task.priority || ''}, state=${task.state}`,
  priority: task.priority || '',
  assignee: '',
  edges: [],
});

export const TaskRagPanel: React.FC<TaskRagPanelProps> = ({
  task,
  userId = 'demo-user-uid',
}) => {
  const [snapshot, setSnapshot] = useState(() => readSnapshot(userId));
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState<RAGResult | null>(null);
  const [notice, setNotice] = useState('');
  const [ragBusy, setRagBusy] = useState(false);
  const [ragStage, setRagStage] = useState('');
  const [prediction, setPrediction] = useState<TaskPrediction | null>(() =>
    predictTask(userId, {
      title: task.title,
      description: task.description,
      priority: task.priority,
      category: task.category,
      startDate: task.startDate,
      dueDate: task.dueDate,
      dueTime: task.dueTime,
    })
  );

  useEffect(() => {
    setSnapshot(readSnapshot(userId));
    return subscribeSnapshot(userId, () => setSnapshot(readSnapshot(userId)));
  }, [userId]);

  useEffect(() => {
    const refresh = () =>
      setPrediction(
        predictTask(userId, {
          title: task.title,
          description: task.description,
          priority: task.priority,
          category: task.category,
          startDate: task.startDate,
          dueDate: task.dueDate,
          dueTime: task.dueTime,
        })
      );
    void requestRestore(userId);
    refresh();
    return subscribeActiveResult(userId, refresh);
  }, [userId, task]);

  useEffect(() => {
    setAnswer(null);
    setNotice('');
    setQuery('');
    setRagBusy(false);
    setRagStage('');
  }, [task.id]);

  const profile = useMemo(() => snapshot?.profile ?? profileData([], []), [snapshot]);
  const suggestions = snapshot?.result?.suggestions ?? [];
  const docs = useMemo(() => {
    const savedDocs = ragDocs(snapshot?.dataset ?? null, profile, suggestions);
    return [taskToRagDoc(task, prediction), ...savedDocs, ...KNOWLEDGE_DOCS];
  }, [task, prediction, snapshot, profile, suggestions]);
  const facts = useMemo(() => taskFactLines(task, prediction), [task, prediction]);

  const handleAsk = async (event: React.FormEvent) => {
    event.preventDefault();
    const question = query.trim();
    if (!question) {
      setNotice('Type a question about this task or your saved data.');
      return;
    }
    setNotice('');
    setRagBusy(true);
    setRagStage('');
    setAnswer(null);
    await new Promise<void>((resolve) => window.setTimeout(resolve, 30));
    try {
      const retrieved = runRAG(question, USER_RAG_MODE, docs, profile, suggestions);
      setRagStage('Connecting related ideas…');
      const direct = directTaskAnswer(question, task, prediction);
      const isAbout = /\babout\b|\bdescribe\b|\bsummary\b|\bsummarize\b/i.test(question);
      const answerFacts = isAbout || !direct ? facts : [direct, ...facts];
      const next = await polishRagAnswer({
        question,
        rag: retrieved,
        facts: answerFacts,
        onProgress: setRagStage,
      });
      setAnswer({ ...next, answer: stripIdFragments(next.answer) });
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
    <div className="mt-6 pt-5 border-t border-[var(--line)]">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-xs uppercase font-bold tracking-wider text-[var(--muted)] flex items-center gap-1.5">
            <Search className="w-3.5 h-3.5 text-[var(--accent)]" /> Ask about this task
          </h3>
          <p className="text-[11px] text-[var(--muted)] mt-1.5">
            {snapshot?.upload ? (
              <>
                Auto-detected source: <strong className="text-[var(--ink)]">{snapshot.upload.name}</strong>
                {snapshot.sheetName ? ` · ${snapshot.sheetName}` : ''}. This task is included in the index.
              </>
            ) : (
              <>This task is indexed. Upload a file in ML Lab to include your own data in every task RAG.</>
            )}
          </p>
        </div>
        <Badge size="sm" variant="neutral">{docs.length} sources</Badge>
      </div>

      {prediction && (
        <div className="mt-3 flex flex-wrap items-center gap-2 rounded-lg border border-[var(--accent)]/25 bg-[var(--accent-soft)] px-3 py-2 text-[11px]">
          <Sparkles className="w-3.5 h-3.5 text-[var(--accent)]" />
          <span className="font-bold text-[var(--ink)]">
            {prediction.isCompletion ? 'Completion probability' : 'Best-model probability'}
          </span>
          <Badge size="sm" variant="accent">
            {formatTaskProbability(prediction.probability, prediction.kind)}
          </Badge>
          <span className="text-[var(--muted)]">based on your saved data</span>
        </div>
      )}

      <form onSubmit={handleAsk} className="flex flex-col sm:flex-row gap-2 mt-3">
        <input
          aria-label="Ask about this task"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setNotice('');
          }}
          placeholder="Ask anything about this task or your saved data…"
          className="flex-1 px-3 py-2 text-sm bg-[var(--surface)] border border-[var(--line)] rounded-lg"
        />
        <Button type="submit" variant="secondary" size="sm" disabled={ragBusy}>
          {ragBusy ? 'Finding…' : 'Ask'}
        </Button>
      </form>

      <div className="flex flex-wrap gap-1.5 mt-2">
        {[
          'What is this task about?',
          'When is this task due?',
          `What is the completion probability of ${task.title}?`,
          'What should I do next?',
        ].map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => {
              setQuery(example);
              setNotice('');
            }}
            className="text-[10px] px-2 py-1 rounded-full border border-[var(--line)] text-[var(--muted)] hover:bg-[var(--canvas)] cursor-pointer"
          >
            {example}
          </button>
        ))}
      </div>
      {notice && <p className="text-[11px] text-[var(--muted)] mt-2">{notice}</p>}

      {ragBusy && <RagWorking compact stage={ragStage} />}

      {answer && !ragBusy && (
        <div className="mt-3 rounded-lg border border-[var(--line)] bg-[var(--canvas)] p-3">
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
    </div>
  );
};

export default TaskRagPanel;
