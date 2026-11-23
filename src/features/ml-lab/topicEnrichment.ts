import { composeAnswer, type ComposeAnswerOptions } from './answerEngine';
import { topicWords } from './topicLanguage';
import { lookupTopic, lookupTopicWord, readCachedTopic, type TopicSummary } from './topicLookup';

const LOOKUP_TIMEOUT = 1600;

/** Task wording is not a subject, so it should never trigger a Wikipedia search. */
const TASK_WORDS = new Set([
  'about', 'action', 'again', 'also', 'class', 'completion', 'deadline', 'describe', 'details',
  'due', 'explain', 'help', 'latest', 'lesson', 'mark', 'next', 'please', 'planned', 'probability',
  'question', 'schedule', 'scheduled', 'show', 'step', 'task', 'tell', 'today', 'update', 'what',
  'when', 'where', 'which', 'while', 'with', 'your',
]);

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

  const cleanSubject = String(subject || '').trim();
  // Keep real topic words in their original form: "calculus", not the stemmed
  // "calculu" used by the offline matcher.
  const wordQueries = topicWords(cleanSubject).filter(
    (word) => word.length > 2 && !TASK_WORDS.has(word)
  );

  const cached = [cleanSubject, ...wordQueries].map(readCachedTopic).find(Boolean);
  if (cached) {
    apply(cached);
    return () => {
      cancelled = true;
    };
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), LOOKUP_TIMEOUT);

  const run = async () => {
    try {
      const exact = await lookupTopic(cleanSubject, controller.signal);
      if (cancelled || controller.signal.aborted) return;
      if (exact) {
        apply(exact);
        return;
      }

      // Word-by-word fallback: a long task title rarely has its own page, but
      // each meaningful word can still be checked quickly and in parallel.
      const results = await Promise.allSettled(
        wordQueries.map((word) => lookupTopicWord(word, controller.signal))
      );
      if (cancelled || controller.signal.aborted) return;

      const found = results.find(
        (result): result is PromiseFulfilledResult<TopicSummary | null> =>
          result.status === 'fulfilled' && Boolean(result.value)
      )?.value;
      if (found) apply(found);
    } catch {
      // Network is optional: keep the instant local answer.
    }
  };

  void run().finally(() => clearTimeout(timer));
  return () => {
    cancelled = true;
    clearTimeout(timer);
    controller.abort();
  };
}
