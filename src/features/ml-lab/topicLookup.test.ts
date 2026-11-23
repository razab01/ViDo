import { afterEach, describe, expect, it, vi } from 'vitest';
import { lookupTopic, lookupTopicWord, readCachedTopic } from './topicLookup';

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

  it('searches an individual word, fetches the top result and caches it', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const json = url.includes('list=search')
        ? { query: { search: [{ title: 'Calculus' }] } }
        : {
            query: {
              pages: {
                '5176': {
                  title: 'Calculus',
                  extract:
                    'Calculus is the branch of mathematics that studies continuous change, and is the principal precursor of modern mathematical analysis.',
                },
              },
            },
          };
      return { ok: true, json: async () => json } as unknown as Response;
    });
    vi.stubGlobal('fetch', fetchMock);

    const summary = await lookupTopicWord('calculus');
    expect(summary?.title).toBe('Calculus');
    expect(String(fetchMock.mock.calls[0][0])).toContain('list=search');
    expect(String(fetchMock.mock.calls[0][0])).toContain('srsearch=calculus');
    expect(String(fetchMock.mock.calls[1][0])).toContain('titles=Calculus');
    expect(readCachedTopic('calculus')?.title).toBe('Calculus');
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
