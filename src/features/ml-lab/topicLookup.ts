/**
 * Optional live topic enrichment from Wikipedia.
 *
 * ViDo never waits for this: answers are composed first from local word
 * understanding and bundled knowledge. Successful lookups are cached for 30
 * days, so repeat questions are instant and the app still works offline.
 */

export interface TopicSummary {
  title: string;
  extract: string;
}

const CACHE_PREFIX = 'vido:topic:';
const CACHE_TTL = 1000 * 60 * 60 * 24 * 30;

const cacheKey = (subject: string): string =>
  CACHE_PREFIX +
  String(subject || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');

export function readCachedTopic(subject: string): TopicSummary | null {
  const key = cacheKey(subject);
  if (!key || key === CACHE_PREFIX) return null;
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as TopicSummary & { cachedAt?: number };
    if (typeof parsed.extract !== 'string' || !parsed.extract) return null;
    if (parsed.cachedAt && Date.now() - parsed.cachedAt > CACHE_TTL) {
      localStorage.removeItem(key);
      return null;
    }
    return { title: parsed.title || subject, extract: parsed.extract };
  } catch {
    return null;
  }
}

const writeCachedTopic = (subject: string, summary: TopicSummary): void => {
  try {
    localStorage.setItem(
      cacheKey(subject),
      JSON.stringify({ ...summary, cachedAt: Date.now() })
    );
  } catch {
    // Storage may be disabled; the answer still works without a cache.
  }
};

/**
 * Fetch a short Wikipedia summary. Returns null on any network/CORS problem,
 * so callers can keep the locally composed answer.
 */
export async function lookupTopic(subject: string, signal?: AbortSignal): Promise<TopicSummary | null> {
  const title = String(subject || '').trim();
  if (!title) return null;

  const cached = readCachedTopic(title);
  if (cached) return cached;

  const url =
    'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1' +
    '&exsentences=2&redirects=1&format=json&origin=*&titles=' +
    encodeURIComponent(title);

  const response = await fetch(url, { signal, headers: { Accept: 'application/json' } });
  if (!response.ok) return null;

  const payload = (await response.json()) as {
    query?: { pages?: Record<string, { title?: string; extract?: string }> };
  };
  const page = Object.values(payload.query?.pages || {})[0];
  const extract = String(page?.extract || '')
    .replace(/\s+/g, ' ')
    .trim();

  if (extract.length < 40 || /may refer to|could refer to/i.test(extract)) return null;

  const summary: TopicSummary = { title: page?.title || title, extract };
  writeCachedTopic(title, summary);
  return summary;
}

/**
 * Word-level Wikipedia search for long-tail topics.
 *
 * A full title such as "Schedule for calculus class" has no Wikipedia page,
 * so we search for each meaningful word ("calculus"), take the top result and
 * fetch its short summary. The result is cached under the original word.
 */
export async function lookupTopicWord(
  word: string,
  signal?: AbortSignal
): Promise<TopicSummary | null> {
  const query = String(word || '').trim();
  if (!query) return null;

  const cached = readCachedTopic(query);
  if (cached) return cached;

  const searchUrl =
    'https://en.wikipedia.org/w/api.php?action=query&list=search&srnamespace=0' +
    '&srlimit=1&format=json&origin=*&srsearch=' +
    encodeURIComponent(query);

  const response = await fetch(searchUrl, { signal, headers: { Accept: 'application/json' } });
  if (!response.ok) return null;

  const payload = (await response.json()) as {
    query?: { search?: Array<{ title?: string }> };
  };
  const title = payload.query?.search?.find((item) => Boolean(item.title))?.title;
  if (!title) return null;

  const summary = await lookupTopic(title, signal);
  if (summary) writeCachedTopic(query, summary);
  return summary;
}
