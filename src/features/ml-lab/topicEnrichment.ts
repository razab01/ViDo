import { composeAnswer, type ComposeAnswerOptions } from './answerEngine';
import { subjectKeywords } from './topicLanguage';
import { lookupTopic, readCachedTopic, type TopicSummary } from './topicLookup';

const LOOKUP_TIMEOUT = 1600;

/**
 * Improve an already-displayed answer with a cached or freshly fetched
 * Wikipedia summary. The caller shows the local answer immediately; this
 * function never blocks it and returns a cancel function for stale requests.
 */
export function enrichAnswerWithTopic(
  options: ComposeAnswerOptions,
  subject: string,
  currentAnswer: string,
  onUpdate: (answer: string) => void
): () => void {
  let cancelled = false;

  const apply = (topic: TopicSummary) => {
    if (cancelled) return;
    const improved = composeAnswer({ ...options, topic });
    if (improved && improved !== currentAnswer) onUpdate(improved);
  };

  const keywords = subjectKeywords(subject);
  const queries = [subject, ...(keywords.length > 3 ? [keywords.slice(0, 2).join(' ')] : [])].filter(Boolean);

  const cached = queries.map(readCachedTopic).find(Boolean);
  if (cached) {
    apply(cached);
    return () => {
      cancelled = true;
    };
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), LOOKUP_TIMEOUT);

  const run = async () => {
    for (const query of queries) {
      try {
        const topic = await lookupTopic(query, controller.signal);
        if (cancelled || controller.signal.aborted) return;
        if (topic) {
          apply(topic);
          return;
        }
      } catch {
        // Network is optional: keep the instant local answer.
        return;
      }
    }
  };

  void run().finally(() => clearTimeout(timer));
  return () => {
    cancelled = true;
    clearTimeout(timer);
    controller.abort();
  };
}
