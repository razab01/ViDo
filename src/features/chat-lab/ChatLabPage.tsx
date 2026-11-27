import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  ArrowRight,
  FileText,
  Loader2,
  Mic,
  Paperclip,
  Plus,
  Send,
  Sparkles,
  Square,
  X,
} from 'lucide-react';
import { Button, Card } from '../../components/ui';
import { buildTaskFlowchart } from '../../data/flowchart';
import type { SampleTask } from '../../data/sampleTasks';
import { handleTaskDraftMessage } from '../../data/taskCreation';
import { loadChatDocument, type ChatDocument } from './documentAnswer';
import { FlowchartPreview } from '../workspace/FlowchartPreview';
import { profileData } from '../ml-lab/engine';
import type { Dataset, Profile } from '../ml-lab/engine';
import {
  answerWithPython,
  isPythonAvailable,
  transcribeAudioWithPython,
  type PythonAnswerResult,
} from '../ml-lab/pythonApi';
import {
  getActiveResult,
  readSnapshot,
  subscribeActiveResult,
} from '../ml-lab/mlStore';
import { answerLocally, resolveQuestion, type ChatTurn } from './chatBrain';
import {
  bookingConfirmation,
  isGraphBookingRequest,
  localBookingTasks,
  plannedTaskToSampleTask,
} from './planBooking';
import {
  addMessage,
  createThread,
  ensureActiveThread,
  getActiveChatId,
  patchMessage,
  readThreads,
  setThreadBooking,
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
  'Book a task for tomorrow at 9am',
  'Book the tasks in my attachment',
];

const id = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

const VOICE_MIME_TYPES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/ogg;codecs=opus',
  'audio/mp4',
];

const voiceMimeType = (): string => {
  if (typeof MediaRecorder === 'undefined' || !MediaRecorder.isTypeSupported) return '';
  return VOICE_MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type)) ?? '';
};

const voiceFileExtension = (mime: string): string => {
  if (mime.includes('ogg')) return 'ogg';
  if (mime.includes('mp4')) return 'm4a';
  return 'webm';
};

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
  const [listening, setListening] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  /** Staged in the composer; it is posted only with the next chat message. */
  const [pendingDocument, setPendingDocument] = useState<ChatDocument | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);
  const documentInputRef = useRef<HTMLInputElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    const unsubscribeChats = subscribeThreads(userId, () => {
      setThreads(readThreads(userId));
      setActiveId(getActiveChatId(userId));
    });
    const unsubscribeModel = subscribeActiveResult(userId, setActive);

    // Subscribe first so a thread created on this first render is reflected in
    // state even if someone types before the initial effect settles.
    ensureActiveThread(userId);
    setThreads(readThreads(userId));
    setActiveId(getActiveChatId(userId));

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

  useEffect(
    () => () => {
      const recorder = mediaRecorderRef.current;
      if (recorder) {
        recorder.onstop = null;
        if (recorder.state !== 'inactive') recorder.stop();
      }
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      mediaRecorderRef.current = null;
      mediaStreamRef.current = null;
    },
    []
  );

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
    if (!question || busy) return;
    const attachedDocument = pendingDocument;
    setPendingDocument(null);

    // A fast click can arrive before the thread effect has published state.
    // Never drop the message silently in that case.
    let currentThread = thread;
    if (!currentThread) {
      currentThread = ensureActiveThread(userId);
      setThreads(readThreads(userId));
      setActiveId(currentThread.id);
    }

    const history = turnsFrom(messages);
    const resolved = resolveQuestion(question, history);
    const graphBookingRequest = isGraphBookingRequest(
      resolved,
      Boolean(attachedDocument ?? currentThread.document)
    );
    const userMessage: ChatMessage = {
      id: id(),
      role: 'user',
      text: question,
      ...(attachedDocument ? { attachment: attachedDocument } : {}),
    };
    const pendingMessage: ChatMessage = { id: id(), role: 'assistant', text: '', pending: true };
    addMessage(userId, currentThread.id, userMessage);
    if (attachedDocument) {
      setThreadDocument(userId, currentThread.id, attachedDocument);
      // A booking turn should go straight to the plan; the "Loaded…" aside is
      // only useful when the user is asking about the document itself.
      if (!graphBookingRequest) {
        addMessage(userId, currentThread.id, {
          id: id(),
          role: 'assistant',
          text: `Loaded "${attachedDocument.name}". Ask me anything about it.`,
        });
      }
    }
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

      // A booking request belongs to Graph RAG even when an attachment is
      // active. Clear any stale draft so follow-ups cannot be pulled back into
      // the old "what should I call it" loop.
      const graphBookingActive = Boolean(currentThread.booking);
      if ((graphBookingRequest || graphBookingActive) && currentThread.taskDraft) {
        setThreadTaskDraft(userId, currentThread.id, null);
      }

      if (onCreateTask && !graphBookingRequest && !graphBookingActive) {
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
      // The callback can run inside answerWithPython; a plain `let` would be
      // narrowed to null by TypeScript because the assignment is indirect.
      const pythonResultRef: { current: PythonAnswerResult | null } = { current: null };
      const pythonAvailable = await isPythonAvailable();

      // Offline booking keeps the same friendly flow for a single described
      // task; document bookings still need the Python extractor.
      if (graphBookingRequest && !pythonAvailable && !graphBookingActive) {
        if (!onCreateTask) {
          patchMessage(userId, currentThread.id, pendingMessage.id, {
            text: 'I found the task details, but task creation is not available in this workspace.',
            pending: false,
          });
          return;
        }
        const localTasks = localBookingTasks(resolved);
        if (!localTasks) {
          patchMessage(userId, currentThread.id, pendingMessage.id, {
            text: 'Tell me what you’d like to book and when — like `book a task for morning walk tomorrow at 4am`.',
            pending: false,
          });
          return;
        }
        const localCreated = localTasks.map((task) => plannedTaskToSampleTask(task));
        localCreated.forEach((task) => {
          onCreateTask(task);
          setThreadLastTask(userId, currentThread.id, task);
        });
        setThreadBooking(userId, currentThread.id, null);
        patchMessage(userId, currentThread.id, pendingMessage.id, {
          text: bookingConfirmation(localCreated),
          pending: false,
        });
        return;
      }

      if (pythonAvailable) {
        try {
          answer = await answerWithPython(
            {
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
              documents: (() => {
                const activeDocument = attachedDocument ?? currentThread.document;
                return activeDocument
                  ? [{ name: activeDocument.name, text: activeDocument.text }]
                  : [];
              })(),
              booking: currentThread.booking ?? null,
            },
            (result) => {
              pythonResultRef.current = result;
            }
          );
        } catch {
          answer = '';
          pythonResultRef.current = null;
        }
      }

      if (!answer) {
        answer =
          (await answerLocally(resolved, {
            dataset,
            profile,
            suggestions,
            document: attachedDocument ?? currentThread.document,
          })) ??
          'I could not find a clear answer to that. Try asking about a task ID, your data, a loaded document, or a topic.';
      }

      const pythonResult = pythonResultRef.current;
      if (pythonResult?.booking) {
        setThreadBooking(userId, currentThread.id, pythonResult.booking);
      }

      // Graph RAG planned a booking. The UI owns the actual task creation, so
      // only report "booked" after every task has been created successfully.
      if (pythonResult?.actions?.type === 'book_plans') {
        const actions = pythonResult.actions;
        if (!onCreateTask) {
          setThreadBooking(userId, currentThread.id, null);
          patchMessage(userId, currentThread.id, pendingMessage.id, {
            text: 'I found the plan items, but task creation is not available in this workspace.',
            pending: false,
          });
          return;
        }
        try {
          const created = actions.tasks.map((task) =>
            plannedTaskToSampleTask(task)
          );
          created.forEach((task) => {
            onCreateTask?.(task);
            setThreadLastTask(userId, currentThread.id, task);
          });
          setThreadBooking(userId, currentThread.id, null);
          patchMessage(userId, currentThread.id, pendingMessage.id, {
            text: bookingConfirmation(created),
            pending: false,
          });
        } catch {
          const retryTasks = actions.tasks;
          setThreadBooking(userId, currentThread.id, {
            tasks: retryTasks,
            pending: {
              taskIndex: 0,
              field: 'dueDate',
              question: 'Please give a valid due date for the first plan item.',
            },
          });
          patchMessage(userId, currentThread.id, pendingMessage.id, {
            text: 'One of those plan items has dates I cannot book yet. Please give a valid due date.',
            pending: false,
          });
        }
        return;
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
    setPendingDocument(null);
    setActiveId(null);
  };

  const handleDocumentFile = async (file: File | undefined) => {
    if (!file || !thread || documentBusy) return;
    setDocumentError(null);
    setDocumentBusy(true);
    try {
      const loaded = await loadChatDocument(file);
      setPendingDocument(loaded);
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
    if (pendingDocument) {
      setPendingDocument(null);
      setDocumentError(null);
      return;
    }
    if (thread) setThreadDocument(userId, thread.id, null);
    setDocumentError(null);
  };

  const submitVoiceNote = async (blob: Blob, name: string) => {
    setVoiceError(null);
    setTranscribing(true);
    try {
      const result = await transcribeAudioWithPython(blob, name);
      const transcript = result.text.trim();
      if (!transcript) {
        setVoiceError('I did not catch that. Try your voice note again.');
        return;
      }
      setInput(transcript);
      await ask(transcript);
    } catch (error) {
      const detail = error instanceof Error ? error.message : '';
      setVoiceError(
        detail.includes('not installed') || detail.includes('could not be started')
          ? 'Offline voice is not ready on this device yet.'
          : 'I could not transcribe that voice note. Try again.'
      );
    } finally {
      setListening(false);
      setTranscribing(false);
    }
  };

  const stopVoiceNote = () => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== 'inactive') recorder.stop();
  };

  const startVoiceNote = async () => {
    if (listening) {
      stopVoiceNote();
      return;
    }
    if (busy || transcribing) return;
    setVoiceError(null);

    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setVoiceError('Voice capture is not available in this browser.');
      return;
    }
    const available = await isPythonAvailable();
    if (!available) {
      setVoiceError('Offline voice is not ready on this device yet.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      const mime = voiceMimeType();
      const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      const chunks: Blob[] = [];
      audioChunksRef.current = chunks;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.push(event.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        if (mediaRecorderRef.current === recorder) mediaRecorderRef.current = null;
        mediaStreamRef.current = null;
        const type = recorder.mimeType || mime || 'audio/webm';
        const blob = new Blob(chunks, { type });
        const name = `voice-${Date.now()}.${voiceFileExtension(type)}`;
        void submitVoiceNote(blob, name);
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setListening(true);
    } catch {
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
      mediaRecorderRef.current = null;
      setListening(false);
      setVoiceError('Microphone access was not granted.');
    }
  };

  return (
    <div className="vdb-chat-page p-4 sm:p-6 lg:p-8 max-w-[900px] mx-auto space-y-4">
      <Card className="vdb-chat-card flex flex-col h-[calc(100vh-136px)] min-h-[420px] overflow-hidden">
        <div className="vdb-chat-card-top">
          <div className="vdb-chat-card-heading">
            <span className="vdb-chat-card-label">Conversation</span>
            <span className="vdb-chat-card-hint">Private to this workspace</span>
          </div>
          <Button
            className="vdb-chat-new-btn"
            variant="ghost"
            size="sm"
            onClick={startNewChat}
            disabled={!messages.length && !input}
          >
            <Plus className="w-4 h-4" /> New chat
          </Button>
        </div>
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4" aria-label="Chat messages">
          {messages.length === 0 ? (
            <div className="vdb-chat-empty-state">
              <div className="vdb-chat-empty-mark" aria-hidden="true">
                <Sparkles className="w-7 h-7" />
              </div>
              <p className="vdb-chat-empty-eyebrow">A clear space to think</p>
              <p className="vdb-chat-empty-title">Start a conversation</p>
              <p className="vdb-chat-empty-copy">
                Pick one of these, or type your own.
              </p>
              <div className="vdb-chat-suggestions">
                {SUGGESTIONS.map((suggestion, index) => (
                  <button
                    key={suggestion}
                    type="button"
                    onClick={() => void ask(suggestion)}
                    className="vdb-chat-prompt"
                  >
                    <span className="vdb-chat-prompt-icon" aria-hidden="true">
                      {index === 0 ? (
                        <Sparkles className="w-4 h-4" />
                      ) : (
                        <FileText className="w-4 h-4" />
                      )}
                    </span>
                    <span>{suggestion}</span>
                    <ArrowRight className="w-3.5 h-3.5 vdb-chat-prompt-arrow" aria-hidden="true" />
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((message) =>
              message.role === 'user' ? (
                <div key={message.id} className="flex justify-end">
                  <div className="vdb-chat-bubble-user max-w-[80%] rounded-2xl rounded-br-sm bg-[var(--accent-soft)] px-4 py-2.5 text-sm text-[var(--ink)]">
                    <span className="vdb-chat-bubble-label">You</span>
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
                  <div className="vdb-chat-avatar w-8 h-8 rounded-full bg-[var(--accent)] text-white flex items-center justify-center shrink-0">
                    <Sparkles className="w-4 h-4" />
                  </div>
                  <div className="vdb-chat-bubble-assistant max-w-[85%] rounded-2xl rounded-bl-sm border border-[var(--line)] bg-[var(--surface)] px-4 py-3">
                    <span className="vdb-chat-bubble-label">ViDo</span>
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

        <div className="vdb-chat-composer border-t border-[var(--line)] p-3">
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

          {voiceError && (
            <p
              role="alert"
              className="mb-2 flex items-center gap-1.5 text-xs text-[var(--red)]"
            >
              <AlertCircle className="w-3.5 h-3.5 shrink-0" /> {voiceError}
            </p>
          )}

          {(pendingDocument ?? thread?.document) && (
            <div className="mb-2 inline-flex max-w-full items-center gap-2 rounded-md border border-[var(--line)] bg-[var(--canvas)] px-2.5 py-1.5 text-xs text-[var(--muted)]">
              <FileText className="w-3.5 h-3.5 shrink-0 text-[var(--accent)]" />
              <span
                className="truncate max-w-[240px]"
                title={(pendingDocument ?? thread?.document)?.name}
              >
                {(pendingDocument ?? thread?.document)?.name}
              </span>
              {pendingDocument ? (
                <span className="text-[10px]">· ready to send</span>
              ) : (
                thread?.document?.truncated && <span className="text-[10px]">· shortened</span>
              )}
              <button
                type="button"
                aria-label={pendingDocument ? 'Remove staged document' : 'Remove document'}
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
            placeholder="Chat, book a task, or plan one from an attachment…"
            className="w-full px-3.5 py-2 text-sm bg-[var(--surface)] border border-[var(--line)] rounded-lg resize-none"
          />
          <div className="flex items-center justify-between gap-2 mt-2">
            <div className="flex items-center gap-2 min-w-0">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className={`vdb-voice-btn ${listening ? 'is-listening' : ''}`}
                onClick={() => void startVoiceNote()}
                disabled={busy || transcribing}
                aria-label={listening ? 'Stop voice note' : 'Record voice note'}
                title={
                  listening
                    ? 'Tap to stop and send your voice note.'
                    : 'Record a voice note. Tap again to send it.'
                }
              >
                {transcribing ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : listening ? (
                  <Square className="w-3.5 h-3.5" />
                ) : (
                  <Mic className="w-4 h-4" />
                )}
                <span className="hidden sm:inline">
                  {transcribing ? 'Listening back' : listening ? 'Stop' : 'Voice'}
                </span>
              </Button>
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
