import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  addMessage,
  createThread,
  deleteThread,
  ensureActiveThread,
  getActiveChatId,
  patchMessage,
  readThreads,
  resetChatStore,
  selectThread,
  subscribeThreads,
} from './chatStore';

const USER = 'chat-store-user';

describe('chatStore', () => {
  beforeEach(() => {
    localStorage.clear();
    resetChatStore();
  });

  it('creates and reuses an active thread', () => {
    const first = ensureActiveThread(USER);
    expect(first.title).toBe('New chat');
    expect(getActiveChatId(USER)).toBe(first.id);
    expect(ensureActiveThread(USER).id).toBe(first.id);
    expect(readThreads(USER)).toHaveLength(1);
  });

  it('titles a thread from its first question and persists messages', () => {
    const thread = ensureActiveThread(USER);
    addMessage(USER, thread.id, { id: 'u1', role: 'user', text: 'who is adam smith?' });
    addMessage(USER, thread.id, {
      id: 'a1',
      role: 'assistant',
      text: 'Adam Smith was a Scottish economist.',
    });

    expect(readThreads(USER)[0].title).toBe('who is adam smith?');
    expect(readThreads(USER)[0].messages).toHaveLength(2);

    patchMessage(USER, thread.id, 'a1', { text: 'Adam Smith wrote The Wealth of Nations.' });
    expect(readThreads(USER)[0].messages[1].text).toBe(
      'Adam Smith wrote The Wealth of Nations.'
    );
  });

  it('survives an in-memory reset through localStorage', () => {
    const thread = createThread(USER);
    addMessage(USER, thread.id, { id: 'u1', role: 'user', text: 'what is 2+2' });
    resetChatStore();

    expect(readThreads(USER)[0].messages[0].text).toBe('what is 2+2');
    expect(getActiveChatId(USER)).toBe(thread.id);
  });

  it('selects, creates and deletes threads', () => {
    const first = createThread(USER);
    addMessage(USER, first.id, { id: 'u1', role: 'user', text: 'first chat' });
    const second = createThread(USER);
    addMessage(USER, second.id, { id: 'u2', role: 'user', text: 'second chat' });

    expect(readThreads(USER)[0].id).toBe(second.id);
    expect(getActiveChatId(USER)).toBe(second.id);
    selectThread(USER, first.id);
    expect(getActiveChatId(USER)).toBe(first.id);

    const listener = vi.fn();
    const unsubscribe = subscribeThreads(USER, listener);
    deleteThread(USER, first.id);
    expect(listener).toHaveBeenCalled();
    expect(readThreads(USER).map((thread) => thread.id)).toEqual([second.id]);
    expect(getActiveChatId(USER)).toBe(second.id);
    unsubscribe();
  });
});
