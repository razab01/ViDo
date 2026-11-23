/**
 * ChatLab's offline-capable answer brain.
 *
 * Everything here runs in the browser. Python is still preferred when it is
 * available, but a machine without Python can still ask about world facts,
 * maths, Wikipedia topics, pronouns and the user's own dataset.
 */
import type { Dataset, Profile, Suggestion } from '../ml-lab/engine';
import { lookupTopic, lookupTopicWord } from '../ml-lab/topicLookup';
import { localFallbackAnswer } from './localFallback';

export interface ChatTurn {
  question: string;
  answer: string;
}

interface BrainContext {
  dataset: Dataset | null;
  profile: Profile | null;
  suggestions: Suggestion[];
}

// Pattern order is intent order: "task_text" should beat "has_desc".
const column = (cols: string[], patterns: RegExp[]): string | null => {
  for (const pattern of patterns) {
    const match = cols.find((c) => pattern.test(c.toLowerCase()));
    if (match) return match;
  }
  return null;
};

const SMALL_WORDS = new Set(['of', 'the', 'and', 'in', 'on', 'for', 'de', 'van', 'von']);

/** Keep acronyms as typed while making lowercase subjects readable in replies. */
const titleSubject = (subject: string): string =>
  subject
    .split(/\s+/)
    .map((word, index) => {
      if (!/^[a-z]+$/.test(word)) return word;
      if (index > 0 && SMALL_WORDS.has(word)) return word;
      return word.charAt(0).toUpperCase() + word.slice(1);
    })
    .join(' ');

/** Pull the entity out of "who is X" / "tell me about X" style questions. */
export function extractSubject(question: string): string | null {
  const q = String(question || '').trim();
  const patterns = [
    /^(?:what)\s+did(?:\s+the)?\s+(.+?)\s+(?:write|do|say|discover|invent|create|build|found|publish|develop|contribute)\b.*$/i,
    /^(?:who|what)\s+(?:is|was|are|were)\s+(.+?)[?.!]*$/i,
    /^(?:who's|what's)\s+(.+?)[?.!]*$/i,
    /^(?:tell me about|about|describe|explain|introduce me to|introduce)\s+(.+?)[?.!]*$/i,
  ];
  for (const pattern of patterns) {
    const match = q.match(pattern);
    if (!match) continue;
    const subject = match[1]
      .replace(/^(?:the|a|an)\s+/i, '')
      .replace(/[?.!]+$/, '')
      .trim();
    if (
      subject &&
      subject.length > 1 &&
      !/^(?:he|she|it|him|her|his|they|them|their|this|that)$/i.test(subject)
    ) {
      return titleSubject(subject);
    }
  }
  return null;
}

/** Replace he/she/it with the last entity the user asked about. */
export function resolveQuestion(raw: string, history: ChatTurn[]): string {
  const q = String(raw || '').trim();
  if (!/\b(he|she|it|him|her|his|they|them|their|that person|this person)\b/i.test(q)) return q;

  let subject: string | null = null;
  for (let index = history.length - 1; index >= 0; index -= 1) {
    const found = extractSubject(history[index].question);
    if (found && !/\b(he|she|it|him|her)\b/i.test(found)) {
      subject = found;
      break;
    }
  }
  if (!subject) return q;

  return q
    .replace(/\b(that person|this person|he|she|it)\b/gi, subject)
    .replace(/\b(him|her|them)\b/gi, subject)
    .replace(/\b(his|their)\b/gi, `${subject}'s`);
}

const isCompleted = (value: unknown): boolean =>
  /^(yes|true|1|completed|done|finished|completed on time)$/i.test(String(value ?? '').trim());

const quote = (values: string[]): string => {
  const clean = values.map((value) => value.trim()).filter(Boolean);
  if (clean.length === 0) return '';
  if (clean.length === 1) return `“${clean[0]}”`;
  if (clean.length === 2) return `“${clean[0]}” and “${clean[1]}”`;
  return `${clean.slice(0, -1).map((value) => `“${value}”`).join(', ')} and “${clean[clean.length - 1]}”`;
};

const rowText = (row: Record<string, string>, textCol: string | null, cols: string[]): string =>
  String(
    (textCol && row[textCol]) ||
      cols.map((c) => row[c]).find((value) => Boolean(value && value !== '—')) ||
      'Untitled record'
  );

const rowLabel = (
  row: Record<string, string>,
  idCol: string | null,
  textCol: string | null,
  cols: string[]
): string => {
  const text = rowText(row, textCol, cols);
  const recordId = idCol ? String(row[idCol] ?? '').trim() : '';
  return recordId ? `${recordId} — ${text}` : text;
};

/** Answer dataset questions directly from the rows instead of a vague RAG hit. */
export function datasetAnswer(question: string, context: BrainContext): string | null {
  const { dataset, profile, suggestions } = context;
  if (!dataset || !dataset.rows.length) return null;
  const cols = dataset.cols;
  const lower = question.toLowerCase();

  const textCol =
    (profile?.textCol && cols.includes(profile.textCol) ? profile.textCol : null) ??
    column(cols, [/task_text/, /title/, /desc/, /^text$/, /name/]);
  const idCol =
    column(cols, [/^(task[ _-]?)?id$/, /^id$/]) ??
    (profile?.idLike ?? []).find(
      (c) => cols.includes(c) && c !== textCol && !/text|title|desc|name/i.test(c)
    ) ??
    null;
  const completeCol = column(cols, [/^complet/, /^done$/, /finished/, /^status$/]);
  const dateCol = column(cols, [/due/, /deadline/, /date/]);
  const priorityCol = column(cols, [/priority/]);

  // Exact record lookup: "T09", "tell me about task T09", '"T09" tell me about...'.
  const tokens = question.match(/[A-Za-z]*-?\d+/g) ?? [];
  if (idCol) {
    for (const token of tokens) {
      const match = dataset.rows.find(
        (row) => String(row[idCol] ?? '').trim().toLowerCase() === token.toLowerCase()
      );
      if (!match) continue;
      const parts = [`${token} — ${rowText(match, textCol, cols)}.`];
      if (completeCol) parts.push(`Status: ${String(match[completeCol] || 'unknown')}.`);
      if (dateCol) parts.push(`Due: ${String(match[dateCol] || 'not set')}.`);
      if (priorityCol) parts.push(`Priority: ${String(match[priorityCol] || 'not set')}.`);
      const other = cols.filter(
        (c) => c !== idCol && c !== textCol && c !== completeCol && c !== dateCol && c !== priorityCol
      );
      if (other.length) {
        const detail = other
          .slice(0, 3)
          .map((c) => `${c}: ${String(match[c] || '—')}`)
          .join('; ');
        parts.push(detail + '.');
      }
      return parts.join(' ');
    }
  }

  const asksCompleted = /completed|finished|done/.test(lower);
  if (asksCompleted && /which|what|list|show|most|mostly|many|how much|are there|tasks|records/.test(lower)) {
    const completed = completeCol
      ? dataset.rows.filter((row) => isCompleted(row[completeCol]))
      : [];
    if (/how many|how much|count/.test(lower)) {
      return `${completed.length} ${completed.length === 1 ? 'task is' : 'tasks are'} marked completed in your data.`;
    }
    if (completed.length === 0) {
      return 'No tasks in your data are marked completed yet.';
    }
    const titles = completed.slice(0, 6).map((row) => rowLabel(row, idCol, textCol, cols));
    return `The completed ${completed.length === 1 ? 'task is' : 'tasks are'} ${quote(titles)}.`;
  }

  if (/high priority|urgent/.test(lower) && priorityCol) {
    const urgent = dataset.rows.filter((row) => /high|urgent/i.test(String(row[priorityCol] ?? '')));
    if (!urgent.length) return 'No high-priority tasks are in your data.';
    return `The high-priority ${urgent.length === 1 ? 'task is' : 'tasks are'} ${quote(
      urgent.slice(0, 6).map((row) => rowLabel(row, idCol, textCol, cols))
    )}.`;
  }

  if (/pattern|predict|driver|signal|feature|why/.test(lower)) {
    if (suggestions.length && suggestions[0]) {
      const top = suggestions[0];
      return `The strongest available signals come from the task text and deadline fields. The model currently rates “${top.title}” as ${String(top.label || 'unknown')} (${String(top.conf || 'no confidence')}%).`;
    }
    if (completeCol) {
      const finished = dataset.rows.filter((row) => isCompleted(row[completeCol]));
      if (finished.length) {
        return `Completed records tend to have clear task titles and earlier deadlines, for example ${quote(
          finished.slice(0, 3).map((row) => rowLabel(row, idCol, textCol, cols))
        )}.`;
      }
    }
  }

  return null;
}

/** Wikipedia fallback that works directly from the browser (no Python needed). */
export async function wikipediaAnswer(question: string): Promise<string | null> {
  const subject = extractSubject(question);
  if (!subject || /^\d+$/.test(subject) || subject.length < 3) return null;
  try {
    const summary = (await lookupTopic(subject)) ?? (await lookupTopicWord(subject));
    return summary ? summary.extract.replace(/\s+/g, ' ').trim() : null;
  } catch {
    return null;
  }
}

/**
 * Full browser-side answer path. Returns null when nothing reliable is found
 * so the caller can show a friendly "try again" message.
 */
export async function answerLocally(question: string, context: BrainContext): Promise<string | null> {
  const quick = localFallbackAnswer(question);
  if (quick) return quick;

  const dataAnswer = datasetAnswer(question, context);
  if (dataAnswer) return dataAnswer;

  const wiki = await wikipediaAnswer(question);
  if (wiki) return wiki;

  return null;
}
