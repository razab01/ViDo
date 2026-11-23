import { afterEach, describe, expect, it, vi } from 'vitest';
import { lookupTopic, readCachedTopic } from './topicLookup';

const payload = {
  query: {
    pages: {
      '123': {
        title: 'Boating',
        extract: 'Boating is the recreation of travelling on water in a boat for pleasure or sport.',
      },
    },
  },
};

const mockFetch = (ok: boolean) =>
  vi.fn(async () => ({ ok, json: async () => payload }) as unknown as Response);

afterEach(() => {
  vi.unstubAllGlobals();
  Object.keys(localStorage)
    .filter((key) => key.startsWith('vido:topic:'))
    .forEach((key) => localStorage.removeItem(key));
});

describe('topic lookup', () => {
  it('fetches a short summary and caches it for instant repeat answers', async () => {
    const fetchMock = mockFetch(true);
    vi.stubGlobal('fetch', fetchMock);

    const first = await lookupTopic('Boating');
    expect(first?.extract).toMatch(/travelling on water/i);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    const second = await lookupTopic('Boating');
    expect(second).toEqual(first);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(readCachedTopic('Boating')?.title).toBe('Boating');
  });

  it('returns null when the network is unavailable so local answers still work', async () => {
    vi.stubGlobal('fetch', mockFetch(false));
    expect(await lookupTopic('Unknown subject')).toBeNull();
  });

  it('rejects disambiguation pages', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        ({
          ok: true,
          json: async () => ({
            query: { pages: { '1': { title: 'Mercury', extract: 'Mercury may refer to several things.' } } },
          }),
        }) as unknown as Response
      )
    );
    expect(await lookupTopic('Mercury')).toBeNull();
  });
});
