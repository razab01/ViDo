import type { RAGResult } from './engine';
import { findKnowledge, type KnowledgeEntry } from './knowledge';

export type AnswerProgress = (stage: string) => void;

export interface ComposeAnswerOptions {
  question: string;
  rag: RAGResult;
  facts?: string[];
  onProgress?: AnswerProgress;
}

const MAX_ANSWER_CHARS = 520;

/** Remove identifier fields that should never appear in a user-facing answer. */
const stripIds = (value: string): string =>
  String(value || '')
    .replace(/\b(?:task[_-]?id|source[_-]?row|row[_-]?id|index)\s*:\s*[A-Za-z0-9_-]+\b/gi, '')
    .replace(/\s{2,}/g, ' ')
    .trim();

const findFact = (facts: string[], label: string): string => {
  const source = stripIds(facts.join(' '));
  // Stop at a real sentence end, not at a decimal point inside a percentage.
  const match = new RegExp(`${label}\\s*:\\s*(.+?)(?:\\.(?=\\s|$)|[!?]|$)`, 'i').exec(source);
  return match?.[1]?.trim() || '';
};

const lowerFirst = (value: string): string => {
  const clean = value.replace(/\.$/, '').trim();
  return clean ? clean.charAt(0).toLowerCase() + clean.slice(1) : '';
};

const sentencesOf = (value: string): string[] =>
  String(value || '')
    .match(/[^.!?]+[.!?]+|[^.!?]+$/g)
    ?.map((part) => part.trim())
    .filter(Boolean) || [];

const firstSentence = (value: string): string => sentencesOf(value)[0] || '';
const firstTwoSentences = (value: string): string => sentencesOf(value).slice(0, 2).join(' ');

/** A raw `Label: value` record is never an acceptable answer. */
const isRawField = (value: string): boolean =>
  /^(?:task[_-]?id|task title|description|state|category|priority|start date|due date|due time|due|created|checkpoint|completion probability|estimated outcome|estimated chance|label|confidence|score|title|excerpt|facts to use|relevant records)\s*:/i.test(
    value.trim()
  );

const hasNoContext = (value: string): boolean => /no relevant context|could not find/i.test(value);

const tidy = (parts: string[]): string => {
  const kept: string[] = [];
  let length = 0;
  for (const raw of parts) {
    const part = stripIds(raw).trim();
    if (!part || isRawField(part)) continue;
    const withStop = /[.!?]$/.test(part) ? part : `${part}.`;
    if (length + withStop.length > MAX_ANSWER_CHARS && kept.length) break;
    kept.push(withStop);
    length += withStop.length + 1;
  }
  return kept.join(' ');
};

/** Quote a description so any wording reads naturally in a sentence. */
const detailSentence = (description: string): string =>
  `It is described as “${description.replace(/\.$/, '').trim()}”.`;

const listPhrase = (values: string[]): string => {
  const clean = values.map((value) => stripIds(value).trim()).filter(Boolean).slice(0, 3);
  if (!clean.length) return '';
  if (clean.length === 1) return `“${clean[0]}”`;
  if (clean.length === 2) return `“${clean[0]}” and “${clean[1]}”`;
  return `${clean.slice(0, -1).map((v) => `“${v}”`).join(', ')} and “${clean[clean.length - 1]}”`;
};

type Intent = 'probability' | 'due' | 'nextStep' | 'about' | 'compare' | 'general';

const detectIntent = (question: string): Intent => {
  const q = question.toLowerCase();
  if (/probab|likelihood|chance|confidence|\bscore\b|how likely/.test(q)) return 'probability';
  if (/\bwhen\b|\bdue\b|deadline|by when|\bdate\b|\btime\b/.test(q)) return 'due';
  if (/next step|what should i do|what do i do|recommend|suggest|\baction\b|\bstart\b/.test(q)) return 'nextStep';
  if (/\babout\b|describe|summari[sz]e|explain|what is|what's|\btell me\b/.test(q)) return 'about';
  if (/compare|comparison|trend|pattern|difference|versus|\bvs\b|\bmost\b|\btop\b|likely|predict|\bwhich\b/.test(q))
    return 'compare';
  return 'general';
};

/** Knowledge query that matches the intent, so probability answers explain probability. */
const INTENT_KNOWLEDGE: Partial<Record<Intent, string>> = {
  probability: 'probability confidence prediction chance',
  due: 'deadline scheduling due date time',
  nextStep: 'next action prioritisation plan',
  compare: 'trend pattern compare features categories',
};

const subjectOf = (question: string, facts: string[], rag: RAGResult): string =>
  [
    findFact(facts, 'Task title'),
    findFact(facts, 'Description'),
    findFact(facts, 'Category'),
    question,
    rag.ranked.slice(0, 2).map((item) => item.title).join(' '),
  ]
    .filter(Boolean)
    .join(' ');

const knowledgeFor = (intent: Intent, subject: string, question: string): KnowledgeEntry | null => {
  const byIntent = INTENT_KNOWLEDGE[intent];
  return (byIntent ? findKnowledge(byIntent) : null) || findKnowledge(subject) || findKnowledge(question);
};

const dueParts = (facts: string[], knowledge: KnowledgeEntry | null, question: string): string[] => {
  const direct = findFact(facts, 'Due');
  const date = findFact(facts, 'Due date');
  const time = findFact(facts, 'Due time');
  const parts: string[] = [];

  if (direct) parts.push(`This task is due ${direct}.`);
  else if (date) parts.push(`This task is due ${date}${time ? ` at ${time}` : ''}.`);
  else if (time) parts.push(`The due time is ${time}.`);
  else parts.push('I do not have a due date for this item yet.');

  if (knowledge) parts.push(firstSentence(knowledge.text));
  else if (hasNoContext(question)) parts.push('Add a due date to your data and I can give you the exact deadline.');
  return parts;
};

const probabilityParts = (facts: string[], knowledge: KnowledgeEntry | null): string[] => {
  const completion = findFact(facts, 'Completion probability');
  const outcome = findFact(facts, 'Estimated outcome');
  const generic = findFact(facts, 'Probability') || findFact(facts, 'Chance') || findFact(facts, 'Confidence');
  const parts: string[] = [];

  if (completion) parts.push(`The estimated completion probability is ${completion}.`);
  else if (outcome) parts.push(`The best model estimates ${outcome}.`);
  else if (generic) parts.push(`The estimated chance is ${generic}.`);
  else parts.push('There is no trained probability for this item yet.');

  if (knowledge) parts.push(firstSentence(knowledge.text));
  return parts;
};

const nextStepParts = (
  facts: string[],
  rag: RAGResult,
  knowledge: KnowledgeEntry | null
): string[] => {
  const title = findFact(facts, 'Task title') || rag.ranked[0]?.title || '';
  const priority = findFact(facts, 'Priority');
  const due = findFact(facts, 'Due') || findFact(facts, 'Due date');
  const checkpoint = findFact(facts, 'Checkpoint');
  const parts: string[] = [];

  if (title) {
    parts.push(
      `The next step is to move ${listPhrase([title])} forward${
        priority ? `, starting with the ${priority.toLowerCase()} priority work` : ''
      }${due ? ` and finishing before ${due}` : ''}.`
    );
  } else {
    parts.push('Pick the most important open item and turn it into one concrete action.');
  }
  if (checkpoint) parts.push(`The latest update was ${lowerFirst(checkpoint)}.`);
  if (knowledge) parts.push(firstSentence(knowledge.text));
  return parts;
};

const aboutParts = (
  facts: string[],
  rag: RAGResult,
  knowledge: KnowledgeEntry | null,
  intentKnowledge: boolean
): string[] => {
  const title = findFact(facts, 'Task title') || rag.ranked[0]?.title || '';
  const description = findFact(facts, 'Description');
  const category = findFact(facts, 'Category');
  const priority = findFact(facts, 'Priority');
  const due = findFact(facts, 'Due') || findFact(facts, 'Due date');
  const checkpoint = findFact(facts, 'Checkpoint');
  const parts: string[] = [];

  if (knowledge && !intentKnowledge) parts.push(firstSentence(knowledge.text));
  else if (title)
    parts.push(
      `${listPhrase([title])} is ${category ? `a ${category.toLowerCase()} ` : ''}item about getting this work organised and finished.`
    );

  if (description) parts.push(detailSentence(description));
  if (priority) parts.push(`It is marked ${priority.toLowerCase()} priority.`);
  if (due) parts.push(`It is planned for ${due}.`);
  if (checkpoint) parts.push(`The latest update was ${lowerFirst(checkpoint)}.`);

  if (!parts.length && knowledge) parts.push(firstTwoSentences(knowledge.text));
  return parts;
};

const compareParts = (rag: RAGResult, knowledge: KnowledgeEntry | null): string[] => {
  const titles = rag.ranked.slice(0, 3).map((item) => item.title);
  const parts: string[] = [];
  if (titles.length) parts.push(`The closest records in your data are ${listPhrase(titles)}.`);
  if (knowledge) parts.push(firstTwoSentences(knowledge.text));
  if (!parts.length)
    parts.push('Upload a file or ask about a specific topic and I can compare the records and highlight the pattern.');
  return parts;
};

const generalParts = (
  facts: string[],
  rag: RAGResult,
  knowledge: KnowledgeEntry | null
): string[] => {
  const parts: string[] = [];
  if (knowledge) parts.push(firstTwoSentences(knowledge.text));
  else if (facts.length) {
    const title = findFact(facts, 'Task title');
    const description = findFact(facts, 'Description');
    if (title) parts.push(`${listPhrase([title])} is the item you are asking about.`);
    if (description) parts.push(detailSentence(description));
  } else if (rag.ranked.length) {
    parts.push(`I found related records: ${listPhrase(rag.ranked.slice(0, 3).map((item) => item.title))}.`);
    parts.push('Ask about one of them for a fuller explanation.');
  } else {
    parts.push(
      'I do not have a clear answer for that yet. Try asking about a topic such as prioritisation, deadlines or a task in your data, and I will explain it.'
    );
  }
  return parts;
};

/**
 * Compose a natural answer from Graph RAG results, the caller's structured
 * facts and ViDo's bundled knowledge. No model call is involved, so the reply
 * is immediate and identical for the same question.
 */
export function composeAnswer({ question, rag, facts = [] }: ComposeAnswerOptions): string {
  const cleanQuestion = String(question || '').trim();
  if (!cleanQuestion) return '';

  const safeFacts = facts.map(stripIds).filter(Boolean);
  const intent = detectIntent(cleanQuestion);
  const subject = subjectOf(cleanQuestion, safeFacts, rag);
  // Prefer the item's own name over incidental words in its description or in
  // the question, so a "Visit parents" note cannot outrank a "Boating" task.
  const primarySubject = findFact(safeFacts, 'Task title') || rag.ranked[0]?.title || subject;
  const knowledge = knowledgeFor(intent, primarySubject, cleanQuestion);
  const intentKnowledge = Boolean(
    INTENT_KNOWLEDGE[intent] && knowledge && findKnowledge(primarySubject) !== knowledge
  );

  let parts: string[];
  switch (intent) {
    case 'probability':
      parts = probabilityParts(safeFacts, knowledge);
      break;
    case 'due':
      parts = dueParts(safeFacts, knowledge, cleanQuestion);
      break;
    case 'nextStep':
      parts = nextStepParts(safeFacts, rag, knowledge);
      break;
    case 'about':
      parts = aboutParts(safeFacts, rag, knowledge, intentKnowledge);
      break;
    case 'compare':
      parts = compareParts(rag, knowledge);
      break;
    default:
      parts = generalParts(safeFacts, rag, knowledge);
  }

  const answer = tidy(parts);
  return answer || 'I could not find a clear answer to that. Try asking with different words.';
}

/** Backwards-compatible name used by callers that want a pure fallback. */
export const friendlyFallback = (question: string, rag: RAGResult, facts: string[] = []): string =>
  composeAnswer({ question, rag, facts });

/** Deterministic wrapper kept async so UI callers can show a short status. */
export async function polishRagAnswer(options: ComposeAnswerOptions): Promise<RAGResult> {
  options.onProgress?.('Connecting related ideas…');
  const answer = composeAnswer(options);
  options.onProgress?.('Writing your answer…');
  return { ...options.rag, answer };
}
