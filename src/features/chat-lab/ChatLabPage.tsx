import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  FileText,
  Loader2,
  MessageSquare,
  Paperclip,
  Plus,
  Send,
  Sparkles,
  X,
} from 'lucide-react';
import { Button, Card } from '../../components/ui';
import { buildTaskFlowchart } from '../../data/flowchart';
import type { SampleTask } from '../../data/sampleTasks';
import { handleTaskDraftMessage } from '../../data/taskCreation';
import { loadChatDocument } from './documentAnswer';
import { FlowchartPreview } from '../workspace/FlowchartPreview';
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
  setThreadDocument,
  setThreadLastTask,
  setThreadTaskDraft,
  subscribeThreads,
} from './chatStore';
import type { ChatMessage, ChatThread } from './chatStore';

export interface ChatLabPageProps {
  userId?: string;
  onCreateTask?: (task: SampleTask) => void;
  onUpdateTask?: (task: SampleTask) => void;
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

const GRAPH_REQUEST_RE =
  /\b(?:process\s+graph|process\s+diagram|flowchart|flow\s*chart|workflow|diagram)\b/i;
const GRAPH_REFERENCE_RE = /\b(?:this|it|that|last|previous|current)\b|just\s+made/i;

const wantsLastTaskGraph = (text: string): boolean =>
  GRAPH_REQUEST_RE.test(text) && GRAPH_REFERENCE_RE.test(text);

const formatBytes = (bytes: number): string => {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 KB';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const documentTypeLabel = (name: string): string => {
  const extension = name.toLowerCase().split('.').pop() ?? '';
  const labels: Record<string, string> = {
    pdf: 'PDF',
    docx: 'Word',
    txt: 'Text',
    md: 'Markdown',
    markdown: 'Markdown',
    csv: 'CSV',
    json: 'JSON',
  };
  return labels[extension] ?? 'Document';
};

export const ChatLabPage: React.FC<ChatLabPageProps> = ({
  userId = 'demo-user-uid',
  onCreateTask,
  onUpdateTask,
}) => {
  const [initialSnapshot] = useState(() => readSnapshot(userId));
  const [active, setActive] = useState(() => getActiveResult(userId));
  const [threads, setThreads] = useState<ChatThread[]>(() => readThreads(userId));
  const [activeId, setActiveId] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [documentBusy, setDocumentBusy] = useState(false);
  const [documentError, setDocumentError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);
  const documentInputRef = useRef<HTMLInputElement | null>(null);

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
  const visibleSuggestions = onCreateTask
    ? [...SUGGESTIONS, 'create a task to review payroll tomorrow']
    : SUGGESTIONS;

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
      if (
        onCreateTask &&
        currentThread.lastTask &&
        !currentThread.taskDraft &&
        wantsLastTaskGraph(question)
      ) {
        const flowchart = buildTaskFlowchart(currentThread.lastTask.title);
        const updatedTask: SampleTask = { ...currentThread.lastTask, flowchart };
        setThreadLastTask(userId, currentThread.id, updatedTask);
        onUpdateTask?.(updatedTask);
        patchMessage(userId, currentThread.id, pendingMessage.id, {
          text: `Here's a process graph for "${currentThread.lastTask.title}".`,
          pending: false,
          flowchart,
        });
        return;
      }

      if (onCreateTask) {
        const taskResult = handleTaskDraftMessage(question, currentThread.taskDraft ?? null);
        if (taskResult.handled) {
          setThreadTaskDraft(userId, currentThread.id, taskResult.draft);
          if (taskResult.task) {
            onCreateTask(taskResult.task);
            setThreadLastTask(userId, currentThread.id, taskResult.task);
          }
          patchMessage(userId, currentThread.id, pendingMessage.id, {
            text: taskResult.reply,
            pending: false,
            ...(taskResult.flowchart ? { flowchart: taskResult.flowchart } : {}),
          });
          return;
        }
      }

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
            documents: currentThread.document
              ? [{ name: currentThread.document.name, text: currentThread.document.text }]
              : [],
          });
        } catch {
          answer = '';
        }
      }

      if (!answer) {
        answer =
          (await answerLocally(resolved, {
            dataset,
            profile,
            suggestions,
            document: currentThread.document,
          })) ??
          'I could not find a clear answer to that. Try asking about a task ID, your data, a loaded document, or a topic.';
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
    setDocumentError(null);
    setActiveId(null);
  };

  const handleDocumentFile = async (file: File | undefined) => {
    if (!file || !thread || documentBusy) return;
    setDocumentError(null);
    setDocumentBusy(true);
    try {
      const loaded = await loadChatDocument(file);
      setThreadDocument(userId, thread.id, loaded);
      addMessage(userId, thread.id, {
        id: id(),
        role: 'user',
        text: `Attached "${loaded.name}"`,
        attachment: loaded,
      });
      addMessage(userId, thread.id, {
        id: id(),
        role: 'assistant',
        text: `Loaded "${loaded.name}". Ask me anything about it.`,
      });
    } catch (error) {
      setDocumentError(
        error instanceof Error ? error.message : 'I could not load that document.'
      );
    } finally {
      setDocumentBusy(false);
      if (documentInputRef.current) documentInputRef.current.value = '';
    }
  };

  const removeDocument = () => {
    if (!thread) return;
    setThreadDocument(userId, thread.id, null);
    setDocumentError(null);
  };

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-[900px] mx-auto space-y-4">
      <header className="flex items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-[var(--ink)] flex items-center gap-2">
            <MessageSquare className="w-6 h-6 text-[var(--accent)]" /> Chat Lab
          </h1>
          <p className="text-sm text-[var(--muted)] mt-1">
            Ask anything — maths, world facts, your data, a loaded document, or create a task.
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
                {visibleSuggestions.map((suggestion) => (
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
                    <p>{message.text}</p>
                    {message.attachment && (
                      <div className="mt-2 flex items-center gap-2.5 rounded-lg border border-[var(--line)] bg-[var(--surface)] px-3 py-2">
                        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-[var(--accent)] text-white">
                          <FileText className="w-4 h-4" />
                        </span>
                        <div className="min-w-0">
                          <p className="truncate text-xs font-semibold text-[var(--ink)]" title={message.attachment.name}>
                            {message.attachment.name}
                          </p>
                          <p className="text-[10px] text-[var(--faint)]">
                            {documentTypeLabel(message.attachment.name)} · {formatBytes(message.attachment.size)} · text extracted
                          </p>
                        </div>
                      </div>
                    )}
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
                        {message.flowchart && (
                          <FlowchartPreview
                            flowchart={message.flowchart}
                            className="mt-3 min-w-[320px]"
                          />
                        )}
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
          <input
            ref={documentInputRef}
            type="file"
            accept=".txt,.md,.markdown,.csv,.json,.pdf,.docx"
            aria-label="Choose document"
            className="hidden"
            onChange={(event) => void handleDocumentFile(event.target.files?.[0])}
          />

          {documentError && (
            <p
              role="alert"
              className="mb-2 flex items-center gap-1.5 text-xs text-[var(--red)]"
            >
              <AlertCircle className="w-3.5 h-3.5 shrink-0" /> {documentError}
            </p>
          )}

          {thread?.document && (
            <div className="mb-2 inline-flex max-w-full items-center gap-2 rounded-md border border-[var(--line)] bg-[var(--canvas)] px-2.5 py-1.5 text-xs text-[var(--muted)]">
              <FileText className="w-3.5 h-3.5 shrink-0 text-[var(--accent)]" />
              <span className="truncate max-w-[240px]" title={thread.document.name}>
                {thread.document.name}
              </span>
              {thread.document.truncated && <span className="text-[10px]">· shortened</span>}
              <button
                type="button"
                aria-label="Remove document"
                onClick={removeDocument}
                className="rounded p-0.5 hover:bg-[var(--surface)] text-[var(--faint)] hover:text-[var(--ink)]"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          )}

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
            placeholder="Ask anything or create a task…"
            className="w-full px-3.5 py-2 text-sm bg-[var(--surface)] border border-[var(--line)] rounded-lg resize-none"
          />
          <div className="flex items-center justify-between gap-2 mt-2">
            <div className="flex items-center gap-2 min-w-0">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => documentInputRef.current?.click()}
                disabled={documentBusy || busy}
                title="Attach a PDF, Word, TXT, MD, CSV or JSON file. Images are not allowed."
              >
                {documentBusy ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Paperclip className="w-4 h-4" />
                )}
                Attach document
              </Button>
              <span className="text-[10px] text-[var(--faint)] truncate hidden sm:inline">
                Enter to send · Shift+Enter for a new line
              </span>
            </div>
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
