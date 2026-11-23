import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Loader2, MessageSquare, Plus, Send, Sparkles } from 'lucide-react';
import { Button, Card } from '../../components/ui';
import { profileData } from '../ml-lab/engine';
import type { Dataset, Profile } from '../ml-lab/engine';
import { answerWithPython, isPythonAvailable } from '../ml-lab/pythonApi';
import {
  getActiveResult,
  readSnapshot,
  subscribeActiveResult,
} from '../ml-lab/mlStore';
import { answerLocally, resolveQuestion, type ChatTurn } from './chatBrain';
import {
  addMessage,
  createThread,
  ensureActiveThread,
  getActiveChatId,
  patchMessage,
  readThreads,
  subscribeThreads,
} from './chatStore';
import type { ChatMessage, ChatThread } from './chatStore';

export interface ChatLabPageProps {
  userId?: string;
}

const SUGGESTIONS = [
  'who is president of spain',
  'what is 2+2',
  'Tell me about India',
  'which tasks are completed?',
];

const id = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

const turnsFrom = (messages: ChatMessage[]): ChatTurn[] => {
  const turns: ChatTurn[] = [];
  for (let index = 0; index < messages.length - 1; index += 1) {
    if (messages[index].role === 'user' && messages[index + 1].role === 'assistant') {
      turns.push({ question: messages[index].text, answer: messages[index + 1].text });
      index += 1;
    }
  }
  return turns;
};

export const ChatLabPage: React.FC<ChatLabPageProps> = ({ userId = 'demo-user-uid' }) => {
  const [initialSnapshot] = useState(() => readSnapshot(userId));
  const [active, setActive] = useState(() => getActiveResult(userId));
  const [threads, setThreads] = useState<ChatThread[]>(() => readThreads(userId));
  const [activeId, setActiveId] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    ensureActiveThread(userId);
    const unsubscribeChats = subscribeThreads(userId, () => {
      setThreads(readThreads(userId));
      setActiveId(getActiveChatId(userId));
    });
    const unsubscribeModel = subscribeActiveResult(userId, setActive);
    return () => {
      unsubscribeChats();
      unsubscribeModel();
    };
  }, [userId]);

  useEffect(() => {
    setThreads(readThreads(userId));
    setActiveId(getActiveChatId(userId));
  }, [userId]);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: 'nearest' });
  }, [threads, activeId]);

  const thread = useMemo(
    () => threads.find((item) => item.id === (activeId ?? threads[0]?.id)) ?? null,
    [threads, activeId]
  );
  const messages = thread?.messages ?? [];

  const dataset: Dataset | null = active?.dataset ?? initialSnapshot?.dataset ?? null;
  const profile: Profile = active?.profile ?? initialSnapshot?.profile ?? profileData([], []);
  const suggestions = active?.suggestions ?? [];

  const ask = async (raw: string) => {
    const question = raw.trim();
    if (!question || busy || !thread) return;

    const currentThread = thread;
    const history = turnsFrom(messages);
    const resolved = resolveQuestion(question, history);
    const userMessage: ChatMessage = { id: id(), role: 'user', text: question };
    const pendingMessage: ChatMessage = { id: id(), role: 'assistant', text: '', pending: true };
    addMessage(userId, currentThread.id, userMessage);
    addMessage(userId, currentThread.id, pendingMessage);
    setInput('');
    setBusy(true);

    try {
      let answer = '';

      if (await isPythonAvailable()) {
        try {
          answer = await answerWithPython({
            question: resolved,
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
        } catch {
          answer = '';
        }
      }

      if (!answer) {
        answer =
          (await answerLocally(resolved, { dataset, profile, suggestions })) ??
          'I could not find a clear answer to that. Try asking about a task ID, your data, or a topic.';
      }

      patchMessage(userId, currentThread.id, pendingMessage.id, {
        text: answer.replace(/^Wikipedia:\s*/, '').trim(),
        pending: false,
      });
    } catch {
      patchMessage(userId, currentThread.id, pendingMessage.id, {
        text: 'Sorry, I could not answer that right now. Please try again.',
        pending: false,
      });
    } finally {
      setBusy(false);
    }
  };

  const startNewChat = () => {
    createThread(userId);
    setInput('');
    setBusy(false);
    setActiveId(null);
  };

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-[900px] mx-auto space-y-4">
      <header className="flex items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-[var(--ink)] flex items-center gap-2">
            <MessageSquare className="w-6 h-6 text-[var(--accent)]" /> Chat Lab
          </h1>
          <p className="text-sm text-[var(--muted)] mt-1">
            Ask anything — maths, world facts, your data, or follow up with a pronoun.
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
                      <p className="text-sm text-[var(--ink)] whitespace-pre-wrap">{message.text}</p>
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
