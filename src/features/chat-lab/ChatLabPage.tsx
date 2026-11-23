import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Loader2, MessageSquare, Plus, Send, Sparkles } from 'lucide-react';
import { Badge, Button, Card } from '../../components/ui';
import {
  profileData,
  ragDocs,
  runRAG,
  USER_RAG_MODE,
} from '../ml-lab/engine';
import type { Dataset, Profile, RAGDoc, TrainingResult } from '../ml-lab/engine';
import { KNOWLEDGE_DOCS } from '../ml-lab/knowledge';
import { polishRagAnswer } from '../ml-lab/answerEngine';
import { answerWithPython, isPythonAvailable } from '../ml-lab/pythonApi';
import {
  getActiveResult,
  readSnapshot,
  subscribeActiveResult,
} from '../ml-lab/mlStore';

export interface ChatLabPageProps {
  userId?: string;
}

interface ChatMessage {
  id: number;
  role: 'user' | 'assistant';
  text: string;
  pending?: boolean;
  engine?: 'python' | 'browser' | 'error';
}

const SUGGESTIONS = [
  'who is president of spain',
  'what is eigenvector of [[2,3],[4,5]]',
  'Tell me about India',
  'What patterns predict completed?',
];

const engineBadge = (engine?: ChatMessage['engine']) => {
  if (engine === 'python') return <Badge size="sm" variant="accent">Python</Badge>;
  if (engine === 'browser') return <Badge size="sm" variant="neutral">Local</Badge>;
  if (engine === 'error') return <Badge size="sm" variant="danger">Retry</Badge>;
  return null;
};

export const ChatLabPage: React.FC<ChatLabPageProps> = ({ userId = 'demo-user-uid' }) => {
  const [initialSnapshot] = useState(() => readSnapshot(userId));
  const [active, setActive] = useState<TrainingResult | null>(() => getActiveResult(userId));
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const idRef = useRef(0);
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => subscribeActiveResult(userId, (next) => setActive(next)), [userId]);
  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: 'nearest' });
  }, [messages]);

  const dataset: Dataset | null = active?.dataset ?? initialSnapshot?.dataset ?? null;
  const profile: Profile = active?.profile ?? initialSnapshot?.profile ?? profileData([], []);
  const suggestions = active?.suggestions ?? [];
  const docs = useMemo<RAGDoc[]>(
    () => [...ragDocs(dataset, profile, suggestions), ...KNOWLEDGE_DOCS],
    [dataset, profile, suggestions]
  );

  const ask = async (raw: string) => {
    const question = raw.trim();
    if (!question || busy) return;

    const userMessageId = ++idRef.current;
    const assistantMessageId = ++idRef.current;
    setMessages((current) => [
      ...current,
      { id: userMessageId, role: 'user', text: question },
      { id: assistantMessageId, role: 'assistant', text: '', pending: true },
    ]);
    setInput('');
    setBusy(true);

    try {
      let answer = '';
      let engine: ChatMessage['engine'] = 'browser';

      if (await isPythonAvailable()) {
        try {
          answer = await answerWithPython({
            question,
            rows: dataset?.rows ?? [],
            cols: dataset?.cols ?? [],
            profile,
            suggestions,
            trained: active
              ? {
                  best: active.best,
                  features: active.features,
                  target: active.target,
                  textCol: active.textCol,
                }
              : null,
          });
          engine = 'python';
        } catch {
          answer = '';
        }
      }

      if (!answer) {
        const retrieved = runRAG(question, USER_RAG_MODE, docs, profile, suggestions);
        const polished = await polishRagAnswer({ question, rag: retrieved });
        answer = polished.answer;
        engine = 'browser';
      }

      setMessages((current) =>
        current.map((message) =>
          message.id === assistantMessageId
            ? { ...message, text: answer, pending: false, engine }
            : message
        )
      );
    } catch {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantMessageId
            ? {
                ...message,
                text: 'Sorry, I could not answer that right now. Please try again.',
                pending: false,
                engine: 'error',
              }
            : message
        )
      );
    } finally {
      setBusy(false);
    }
  };

  const startNewChat = () => {
    setMessages([]);
    setInput('');
    setBusy(false);
    idRef.current = 0;
  };

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-[900px] mx-auto space-y-4">
      <header className="flex items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-[var(--ink)] flex items-center gap-2">
            <MessageSquare className="w-6 h-6 text-[var(--accent)]" /> Chat Lab
          </h1>
          <p className="text-sm text-[var(--muted)] mt-1">
            Ask anything — maths, world facts, or your saved data. Answers are routed through Python first.
          </p>
        </div>
        <Button variant="ghost" size="sm" onClick={startNewChat} disabled={!messages.length && !input}>
          <Plus className="w-4 h-4" /> New chat
        </Button>
      </header>

      <Card className="flex flex-col h-[calc(100vh-230px)] min-h-[420px] overflow-hidden">
        <div className="flex-1 overflow-y-auto p-4 space-y-4" aria-label="Chat messages">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center gap-4">
              <div className="w-12 h-12 rounded-full bg-[var(--accent-soft)] flex items-center justify-center">
                <Sparkles className="w-6 h-6 text-[var(--accent)]" />
              </div>
              <div>
                <p className="text-sm font-semibold text-[var(--ink)]">Start a conversation</p>
                <p className="text-xs text-[var(--muted)] mt-1">
                  Try a question below, or type your own.
                </p>
              </div>
              <div className="flex flex-wrap justify-center gap-2 max-w-md">
                {SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    onClick={() => void ask(suggestion)}
                    className="text-[11px] px-3 py-1.5 rounded-full border border-[var(--line)] text-[var(--muted)] hover:bg-[var(--canvas)] cursor-pointer"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((message) =>
              message.role === 'user' ? (
                <div key={message.id} className="flex justify-end">
                  <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-[var(--accent-soft)] px-4 py-2.5 text-sm text-[var(--ink)]">
                    {message.text}
                  </div>
                </div>
              ) : (
                <div key={message.id} className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-full bg-[var(--accent)] text-white flex items-center justify-center shrink-0">
                    <Sparkles className="w-4 h-4" />
                  </div>
                  <div className="max-w-[85%] rounded-2xl rounded-bl-sm border border-[var(--line)] bg-[var(--surface)] px-4 py-3">
                    {message.pending ? (
                      <span className="flex items-center gap-2 text-sm text-[var(--muted)]">
                        <Loader2 className="w-4 h-4 animate-spin text-[var(--accent)]" />
                        Thinking…
                      </span>
                    ) : (
                      <>
                        <p className="text-sm text-[var(--ink)] whitespace-pre-wrap">{message.text}</p>
                        <div className="mt-2">{engineBadge(message.engine)}</div>
                      </>
                    )}
                  </div>
                </div>
              )
            )
          )}
          <div ref={endRef} />
        </div>

        <div className="border-t border-[var(--line)] p-3">
          <textarea
            rows={2}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                void ask(input);
              }
            }}
            placeholder="Ask anything…"
            className="w-full px-3.5 py-2 text-sm bg-[var(--surface)] border border-[var(--line)] rounded-lg resize-none"
          />
          <div className="flex items-center justify-between mt-2">
            <span className="text-[10px] text-[var(--faint)]">
              Enter to send · Shift+Enter for a new line
            </span>
            <Button size="sm" onClick={() => void ask(input)} disabled={busy || !input.trim()}>
              <Send className="w-4 h-4" /> Send
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default ChatLabPage;
