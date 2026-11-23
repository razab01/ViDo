/**
 * Persistent chat history for Chat Lab.
 *
 * Mirrors the mlStore pattern: module-level state with subscriptions so both
 * the chat page and the workspace sidebar stay in sync without prop drilling.
 */

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  pending?: boolean;
}

export interface ChatThread {
  id: string;
  title: string;
  updatedAt: number;
  messages: ChatMessage[];
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
    .map(({ id, role, text }) => ({ id, role, text }));

function persist(userId: string, threads: ChatThread[]) {
  const trimmed = threads.slice(0, MAX_THREADS);
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
          .map((thread) => ({ ...thread, messages: cleanMessages(thread.messages ?? []) }));
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
