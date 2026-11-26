/**
 * Persistent chat history for Converse.
 *
 * Mirrors the mlStore pattern: module-level state with subscriptions so both
 * the chat page and the workspace sidebar stay in sync without prop drilling.
 */

import type { TaskFlowchart } from '../../data/flowchart';
import type { SampleTask } from '../../data/sampleTasks';
import type { TaskDraft } from '../../data/taskCreation';
import type { PythonBookingState } from '../ml-lab/pythonApi';
import type { ChatDocument } from './documentAnswer';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  pending?: boolean;
  flowchart?: TaskFlowchart;
  /** Document posted into the conversation by the user. */
  attachment?: ChatDocument;
}

export interface ChatThread {
  id: string;
  title: string;
  updatedAt: number;
  messages: ChatMessage[];
  taskDraft?: TaskDraft;
  /** Most recent task created from this chat, so "graph for this" has context. */
  lastTask?: SampleTask;
  /** Document loaded in this chat, used as context for answers. */
  document?: ChatDocument;
  /** In-flight Graph RAG plan booking, persisted so clarifications resume. */
  booking?: PythonBookingState | null;
  /** Pinned chats stay at the top of the history list. */
  pinned?: boolean;
}

const STORAGE_PREFIX = 'vido_chat_lab_v1_';
const MAX_THREADS = 30;

const threadsByUser = new Map<string, ChatThread[]>();
const activeByUser = new Map<string, string | null>();
const listeners = new Map<string, Set<() => void>>();

const keyFor = (userId: string) => STORAGE_PREFIX + encodeURIComponent(userId || 'anonymous');

const newId = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

const cleanMessages = (messages: ChatMessage[]): ChatMessage[] =>
  messages
    .filter((message) => !message.pending && Boolean(message.text))
    .map(({ id, role, text, flowchart, attachment }) => ({
      id,
      role,
      text,
      ...(flowchart ? { flowchart } : {}),
      ...(attachment ? { attachment } : {}),
    }));

const documentMime = (name: string): string => {
  const extension = String(name).toLowerCase().split('.').pop() ?? '';
  const types: Record<string, string> = {
    pdf: 'application/pdf',
    docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    txt: 'text/plain',
    md: 'text/markdown',
    markdown: 'text/markdown',
    csv: 'text/csv',
    json: 'application/json',
  };
  return types[extension] ?? 'application/octet-stream';
};

/** Old threads predate the attachment size/mime fields. */
const normalizeDocument = (document: ChatDocument | undefined): ChatDocument | undefined => {
  if (!document) return undefined;
  const estimatedSize = new TextEncoder().encode(document.text || '').length;
  return {
    ...document,
    size: Number.isFinite(document.size) ? document.size : estimatedSize,
    mime: document.mime || documentMime(document.name),
  };
};

/** Older threads loaded a document but never posted it into the conversation. */
const backfillDocumentMessage = (thread: ChatThread): ChatThread => {
  const document = normalizeDocument(thread.document);
  if (!document) return thread;

  const alreadyPosted = (thread.messages ?? []).some(
    (message) => message.attachment?.name === document.name
  );
  if (alreadyPosted) return { ...thread, document };

  const attachmentMessage: ChatMessage = {
    id: newId(),
    role: 'user',
    text: `Attached "${document.name}"`,
    attachment: document,
  };
  const messages = [...(thread.messages ?? [])];
  const loadedIndex = messages.findIndex(
    (message) =>
      message.role === 'assistant' &&
      message.text === `Loaded "${document.name}". Ask me anything about it.`
  );
  if (loadedIndex >= 0) messages.splice(loadedIndex, 0, attachmentMessage);
  else messages.push(attachmentMessage);

  return { ...thread, document, messages };
};

function persist(userId: string, threads: ChatThread[]) {
  // Keep pinned (and the active) conversation safe from the history cap.
  const protectedIds = new Set<string>(
    threads.filter((thread) => thread.pinned).map((thread) => thread.id)
  );
  const activeId = activeByUser.get(userId);
  if (activeId) protectedIds.add(activeId);
  const protectedThreads = threads.filter((thread) => protectedIds.has(thread.id));
  const remainingThreads = threads.filter((thread) => !protectedIds.has(thread.id));
  const trimmed = [...protectedThreads, ...remainingThreads].slice(0, MAX_THREADS);
  threadsByUser.set(userId, trimmed);
  try {
    localStorage.setItem(keyFor(userId), JSON.stringify({ version: 1, threads: trimmed }));
  } catch {
    // Storage may be disabled; the in-memory copy still works for this tab.
  }
  listeners.get(userId)?.forEach((listener) => listener());
}

/** Test helper: forget all in-memory chat state. */
export function resetChatStore() {
  threadsByUser.clear();
  activeByUser.clear();
  listeners.clear();
}

export function readThreads(userId: string): ChatThread[] {
  if (threadsByUser.has(userId)) return threadsByUser.get(userId) ?? [];
  try {
    const raw = localStorage.getItem(keyFor(userId));
    if (raw) {
      const parsed = JSON.parse(raw) as { version?: number; threads?: ChatThread[] };
      if (parsed.version === 1 && Array.isArray(parsed.threads)) {
        const threads = parsed.threads
          .filter((thread) => thread && typeof thread.id === 'string')
          .map((thread) =>
            backfillDocumentMessage({
              ...thread,
              messages: cleanMessages(thread.messages ?? []),
            })
          );
        threadsByUser.set(userId, threads);
        return threads;
      }
    }
  } catch {
    // Corrupt or unavailable storage: start fresh.
  }
  threadsByUser.set(userId, []);
  return [];
}

export function subscribeThreads(userId: string, listener: () => void): () => void {
  const set = listeners.get(userId) ?? new Set();
  set.add(listener);
  listeners.set(userId, set);
  return () => set.delete(listener);
}

export function getActiveChatId(userId: string): string | null {
  if (activeByUser.has(userId)) return activeByUser.get(userId) ?? null;
  const threads = readThreads(userId);
  const id = threads[0]?.id ?? null;
  activeByUser.set(userId, id);
  return id;
}

export function setActiveChatId(userId: string, id: string | null) {
  activeByUser.set(userId, id);
  listeners.get(userId)?.forEach((listener) => listener());
}

export function getActiveThread(userId: string): ChatThread | null {
  const id = getActiveChatId(userId);
  return readThreads(userId).find((thread) => thread.id === id) ?? null;
}

/** Ensure at least one thread exists and return the active one. */
export function ensureActiveThread(userId: string): ChatThread {
  const existing = getActiveThread(userId);
  if (existing) return existing;
  const thread: ChatThread = {
    id: newId(),
    title: 'New chat',
    updatedAt: Date.now(),
    messages: [],
  };
  activeByUser.set(userId, thread.id);
  persist(userId, [thread, ...readThreads(userId)]);
  return thread;
}

export function createThread(userId: string): ChatThread {
  const thread: ChatThread = {
    id: newId(),
    title: 'New chat',
    updatedAt: Date.now(),
    messages: [],
  };
  activeByUser.set(userId, thread.id);
  persist(userId, [thread, ...readThreads(userId)]);
  return thread;
}

export function selectThread(userId: string, id: string) {
  setActiveChatId(userId, id);
}

export function deleteThread(userId: string, id: string) {
  const remaining = readThreads(userId).filter((thread) => thread.id !== id);
  persist(userId, remaining);
  if (getActiveChatId(userId) === id) {
    activeByUser.set(userId, remaining[0]?.id ?? null);
  }
  listeners.get(userId)?.forEach((listener) => listener());
}

const threadTitle = (messages: ChatMessage[]): string => {
  const first = messages.find((message) => message.role === 'user')?.text?.trim() ?? '';
  if (!first) return 'New chat';
  return first.length > 42 ? `${first.slice(0, 42)}…` : first;
};

export function addMessage(userId: string, threadId: string, message: ChatMessage) {
  const threads = readThreads(userId).map((thread) =>
    thread.id === threadId
      ? {
          ...thread,
          title: threadTitle([...thread.messages, message]),
          updatedAt: Date.now(),
          messages: [...thread.messages, message],
        }
      : thread
  );
  persist(userId, threads);
}

export function patchMessage(
  userId: string,
  threadId: string,
  messageId: string,
  patch: Partial<ChatMessage>
) {
  const threads = readThreads(userId).map((thread) =>
    thread.id === threadId
      ? {
          ...thread,
          updatedAt: Date.now(),
          messages: thread.messages.map((message) =>
            message.id === messageId ? { ...message, ...patch } : message
          ),
        }
      : thread
  );
  persist(userId, threads);
}

/** Persist the in-progress conversational task draft for one chat thread. */
export function setThreadTaskDraft(
  userId: string,
  threadId: string,
  draft: TaskDraft | null
) {
  const threads = readThreads(userId).map((thread) =>
    thread.id === threadId
      ? {
          ...thread,
          updatedAt: Date.now(),
          ...(draft ? { taskDraft: draft } : { taskDraft: undefined }),
        }
      : thread
  );
  persist(userId, threads);
}

/** Remember the last task created in a chat for follow-up requests. */
export function setThreadLastTask(
  userId: string,
  threadId: string,
  task: SampleTask | null
) {
  const threads = readThreads(userId).map((thread) =>
    thread.id === threadId
      ? {
          ...thread,
          updatedAt: Date.now(),
          ...(task ? { lastTask: task } : { lastTask: undefined }),
        }
      : thread
  );
  persist(userId, threads);
}

/** Persist an in-flight plan-booking clarification for one chat thread. */
export function setThreadBooking(
  userId: string,
  threadId: string,
  booking: PythonBookingState | null
) {
  const threads = readThreads(userId).map((thread) =>
    thread.id === threadId
      ? {
          ...thread,
          updatedAt: Date.now(),
          ...(booking ? { booking } : { booking: undefined }),
        }
      : thread
  );
  persist(userId, threads);
}

/** Attach or remove the document used as answer context in one chat. */
export function setThreadDocument(
  userId: string,
  threadId: string,
  document: ChatDocument | null
) {
  const threads = readThreads(userId).map((thread) =>
    thread.id === threadId
      ? {
          ...thread,
          updatedAt: Date.now(),
          ...(document ? { document } : { document: undefined }),
        }
      : thread
  );
  persist(userId, threads);
}

/** Pin or unpin a conversation so it stays at the top of the history list. */
export function setThreadPinned(userId: string, threadId: string, pinned: boolean) {
  const threads = readThreads(userId).map((thread) =>
    thread.id === threadId
      ? {
          ...thread,
          ...(pinned ? { pinned: true } : { pinned: undefined }),
        }
      : thread
  );
  persist(userId, threads);
}

export function toggleThreadPinned(userId: string, threadId: string) {
  const thread = readThreads(userId).find((item) => item.id === threadId);
  setThreadPinned(userId, threadId, !thread?.pinned);
}
