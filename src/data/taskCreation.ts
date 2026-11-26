/**
 * Conversational personal-task creation.
 *
 * This module is pure: it parses a user's message into a draft, asks for the
 * next missing detail and finally builds a normal SampleTask. Converse owns the
 * UI side and calls the app's existing task-create callback when a draft is
 * complete.
 */

import {
  getTodayDateString,
  validateChronologicalDeadline,
} from '../utils/dateUtils';
import {
  flowchartToText,
  parseFlowchartSteps,
  validateFlowchart,
  type TaskFlowchart,
} from './flowchart';
import type { SampleTask, TaskCategory, TaskPriority } from './sampleTasks';
import { buildSampleTask } from './taskFactory';

export interface TaskDraft {
  title?: string;
  description?: string;
  startDate?: string;
  dueDate?: string;
  dueTime?: string;
  priority?: TaskPriority;
  category?: TaskCategory;
  wantsFlowchart: boolean;
  /** True once chat has offered (or the user has answered) the flowchart question. */
  flowchartPrompted?: boolean;
  flowchart?: TaskFlowchart;
}

export interface TaskDraftResponse {
  handled: boolean;
  draft: TaskDraft | null;
  reply: string;
  task?: SampleTask;
  flowchart?: TaskFlowchart;
}

const FILLER_PREFIX = /^(?:(?:ok|okay|so|now|and|well|please|hey|hi)\s+)+/i;
const TASK_CREATE_RE =
  /\b(?:create|add|make|new|set\s+up|generate|plan|schedule|remember)\b[^.!?\n]*\b(?:to-?do|todo|task|reminder)\b|\b(?:to-?do|todo|task|reminder)\b[^.!?\n]*\b(?:create|add|make|new|set\s+up|generate|plan|schedule)\b/i;
const FLOWCHART_RE = /\b(?:flowchart|flow\s*chart|process\s+graph|process\s+diagram|diagram|workflow)\b/i;
const CANCEL_RE = /^(?:cancel|forget\s+it|never\s*mind|stop\s*(?:that|it)?|abort)\s*[.!]?$/i;
const SKIP_RE = /^(?:skip|no|none|default|leave\s+it)\s*[.!]?$/i;
const ACCEPT_RE = /^(?:yes|yeah|yep|sure|ok(?:ay)?|please\s+do|i\s+do)\s*[.!]?$/i;
const SKIP_FLOWCHART_RE = /\b(?:skip|without|no|don't\s+want|do\s+not\s+want)\b[^.!?\n]*\b(?:flowchart|flow\s*chart|process\s+graph|process\s+diagram|diagram|workflow)\b|\b(?:flowchart|flow\s*chart|process\s+graph|process\s+diagram|diagram|workflow)\b[^.!?\n]*\b(?:skip|none)\b/i;
const FLOWCHART_PROMPT =
  'Would you like a flowchart for this task? Reply `yes` with steps like `Research -> Build -> Review`, or `no`.';

const WEEKDAYS: Record<string, number> = {
  sun: 0,
  sunday: 0,
  mon: 1,
  monday: 1,
  tue: 2,
  tues: 2,
  tuesday: 2,
  wed: 3,
  weds: 3,
  wednesday: 3,
  thu: 4,
  thur: 4,
  thurs: 4,
  thursday: 4,
  fri: 5,
  friday: 5,
  sat: 6,
  saturday: 6,
};

const MONTHS: Record<string, number> = {
  jan: 0,
  january: 0,
  feb: 1,
  february: 1,
  mar: 2,
  march: 2,
  apr: 3,
  april: 3,
  may: 4,
  jun: 5,
  june: 5,
  jul: 6,
  july: 6,
  aug: 7,
  august: 7,
  sep: 8,
  sept: 8,
  september: 8,
  oct: 9,
  october: 9,
  nov: 10,
  november: 10,
  dec: 11,
  december: 11,
};

const DATE_VALUE =
  '(?:next\\s+\\w+|in\\s+\\d+\\s+days?|\\d{4}-\\d{2}-\\d{2}|\\d{1,2}\\s+[A-Za-z]+(?:\\s+\\d{4})?|[A-Za-z]+)';
const START_LABELS = 'start(?:\\s+date)?|starting|from|begin(?:ning)?(?:\\s+date)?';
const DUE_LABELS = 'due(?:\\s+date)?|deadline|end(?:\\s+date)?|by|until';

const normalize = (raw: string): string => {
  let value = String(raw || '').trim();
  while (FILLER_PREFIX.test(value)) value = value.replace(FILLER_PREFIX, '').trim();
  return value;
};

const formatISO = (date: Date): string => getTodayDateString(date);

const addDays = (now: Date, days: number): Date =>
  new Date(now.getFullYear(), now.getMonth(), now.getDate() + days, 12, 0, 0);

const escapeRegExp = (value: string): string =>
  value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** Convert one user-friendly date token to YYYY-MM-DD. */
export function parseDateToken(raw: string, now: Date = new Date()): string | undefined {
  const value = String(raw || '')
    .trim()
    .replace(/^(?:on|at|for|by|until|from|starting|start(?:\s+date)?|due(?:\s+date)?)\s+/i, '')
    .replace(/[.,]$/, '')
    .toLowerCase();

  if (!value) return undefined;

  const iso = value.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
  if (iso) {
    return `${iso[1]}-${iso[2].padStart(2, '0')}-${iso[3].padStart(2, '0')}`;
  }

  if (/^(?:today|tonight|now|this\s+day)$/.test(value)) return formatISO(now);
  if (/^(?:tomorrow|tmrw|tmr|next\s+day)$/.test(value)) return formatISO(addDays(now, 1));
  if (/^(?:yesterday)$/.test(value)) return formatISO(addDays(now, -1));

  const inDays = value.match(/^in\s+(\d{1,3})\s+days?$/);
  if (inDays) return formatISO(addDays(now, Number(inDays[1])));

  const weekday = value.match(/^(?:next\s+)?([a-z]+)$/);
  if (weekday && weekday[1] in WEEKDAYS) {
    const target = WEEKDAYS[weekday[1]];
    const explicitNext = value.startsWith('next');
    let delta = target - now.getDay();
    if (delta < 0 || (explicitNext && delta === 0)) delta += 7;
    return formatISO(addDays(now, delta));
  }

  const monthDay = value.match(/^([a-z]+)\.?\s+(\d{1,2})(?:,?\s+(\d{4}))?$/);
  if (monthDay && monthDay[1] in MONTHS) {
    const year = monthDay[3] ? Number(monthDay[3]) : now.getFullYear();
    return formatISO(new Date(year, MONTHS[monthDay[1]], Number(monthDay[2]), 12, 0, 0));
  }

  const dayMonth = value.match(/^(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)(?:,?\s+(\d{4}))?$/);
  if (dayMonth && dayMonth[2] in MONTHS) {
    const year = dayMonth[3] ? Number(dayMonth[3]) : now.getFullYear();
    return formatISO(new Date(year, MONTHS[dayMonth[2]], Number(dayMonth[1]), 12, 0, 0));
  }

  return undefined;
}

function extractLabeledDate(text: string, labels: string, now: Date): string | undefined {
  const pattern = new RegExp(
    `\\b(?:${labels})\\b\\s*(?:date)?\\s*(?:is|:|=|on|at)?\\s*(${DATE_VALUE})`,
    'i'
  );
  const match = text.match(pattern);
  return match ? parseDateToken(match[1], now) : undefined;
}

function findBareDate(text: string, now: Date): { value: string; date: string } | null {
  const pattern = new RegExp(
    `\\b(${DATE_VALUE})\\b`,
    'gi'
  );
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text))) {
    const date = parseDateToken(match[1], now);
    if (date) return { value: match[1], date };
  }
  return null;
}

type DayPart = 'morning' | 'afternoon' | 'evening' | 'night' | 'noon' | 'midnight';

function parseTimeToken(raw: string, dayPart?: DayPart): string | undefined {
  const value = String(raw || '')
    .replace(/\bo(?:'|’)?clock\b/gi, '')
    .trim();
  const match = value.match(/\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b/i);
  if (!match) return undefined;

  let hours = Number(match[1]);
  const minutes = match[2] ?? '00';
  let period = match[3]?.toLowerCase();
  if (!period && dayPart) {
    if (dayPart === 'afternoon' || dayPart === 'evening' || dayPart === 'night') period = 'pm';
    if (dayPart === 'morning') period = 'am';
  }
  if (period === 'pm' && hours < 12) hours += 12;
  if (period === 'am' && hours === 12) hours = 0;
  if (dayPart === 'midnight' && hours === 12) hours = 0;
  if (hours > 23 || Number(minutes) > 59) return undefined;
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
}

/** Return the matched wording as well as the normalized time so titles can be cleaned. */
function extractDueTime(text: string): { value: string; time: string } | undefined {
  const named = text.match(/\b(noon|midnight)\b/i);
  if (named) {
    const dayPart = named[1].toLowerCase() as DayPart;
    return { value: named[0], time: parseTimeToken(dayPart === 'noon' ? '12' : '0', dayPart)! };
  }

  const dayPartTime = text.match(
    /\b(morning|afternoon|evening|night)\s+(?:at\s+)?(\d{1,2}(?::\d{2})?\s*(?:o(?:'|’)?clock)?\s*(?:am|pm)?)/i
  );
  if (dayPartTime) {
    const time = parseTimeToken(dayPartTime[2], dayPartTime[1].toLowerCase() as DayPart);
    if (time) return { value: dayPartTime[0], time };
  }

  const labeled = text.match(
    /\b(?:due\s+time|time|at)\s*(?:is|:|=)?\s*(?:in\s+the\s+)?(?:early\s+|late\s+)?(morning|afternoon|evening|night)?\s*(\d{1,2}(?::\d{2})?\s*(?:o(?:'|’)?clock)?\s*(?:am|pm)?)/i
  );
  if (labeled) {
    const time = parseTimeToken(labeled[2], labeled[1]?.toLowerCase() as DayPart | undefined);
    if (time) return { value: labeled[0], time };
  }

  const explicit = text.match(/\b(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b/i);
  if (explicit) {
    const time = parseTimeToken(explicit[1]);
    if (time) return { value: explicit[0], time };
  }
  return undefined;
}

function extractPriority(text: string): TaskPriority | 'clear' | undefined {
  if (/\b(?:no|none|without)\s+priority\b|\bpriority\s*(?:is|:|=)?\s*(?:none|no)\b/i.test(text)) {
    return 'clear';
  }
  if (/\b(?:high|urgent|critical|important)\b/i.test(text)) return 'high';
  if (/\bmedium\b/i.test(text)) return 'medium';
  if (/\blow\b/i.test(text)) return 'low';
  return undefined;
}

function extractCategory(text: string): TaskCategory | undefined {
  if (/\b(?:work|professional|office)\b/i.test(text)) return 'work';
  if (/\b(?:home|household|living)\b/i.test(text)) return 'home';
  if (/\bpersonal\b/i.test(text)) return 'personal';
  return undefined;
}

const stripDirective = (raw: string): string => {
  let value = normalize(raw);
  value = value.replace(/^(?:can|could)\s+you\s+(?:please\s+)?/i, '');
  value = value.replace(
    /^(?:i\s+want(?:\s+to)?|i'd\s+like(?:\s+to)?|i\s+need(?:\s+to)?)\s+/i,
    ''
  );
  value = value.replace(
    /^(?:create|add|make|new|set\s+up|generate|plan|schedule|draw)\s+(?:a|an|the|new)?\s*(?:[a-z]+\s+){0,3}?(?:to-?do|todo|task|reminder|flowchart|flow\s*chart|diagram|workflow)\b\s*(?:for|to|about|called|named|titled|:)?\s*/i,
    ''
  );
  return value.trim();
};

function extractTitle(
  raw: string,
  flowchart: TaskFlowchart | null,
  now: Date,
  timeMatch?: { value: string; time: string }
): string | undefined {
  const quoted = raw.match(/["“'']([^"”'']+)["”'']/);
  if (quoted?.[1]?.trim()) return quoted[1].trim().slice(0, 120);

  let value = stripDirective(raw);
  if (flowchart) {
    const steps = flowchartToText(flowchart);
    if (steps) value = value.split(steps).join(' ');
  }
  value = value.replace(
    /\b(?:with|and|include|including|attach)\s+(?:a\s+)?(?:flowchart|flow\s*chart|diagram|workflow)\b\s*(?:for|of|about|to|:)?\s*/gi,
    ''
  );
  value = value.replace(
    new RegExp(`\\b(?:${START_LABELS})\\b\\s*(?:date)?\\s*(?:is|:|=|on|at)?\\s*${DATE_VALUE}`, 'gi'),
    ' '
  );
  value = value.replace(
    new RegExp(`\\b(?:${DUE_LABELS})\\b\\s*(?:date)?\\s*(?:is|:|=|on|at)?\\s*${DATE_VALUE}`, 'gi'),
    ' '
  );
  value = value.replace(/\b(?:due\s+time|time)\s*(?:is|:|=)?\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b/gi, ' ');
  value = value.replace(/\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b/gi, ' ');
  value = value.replace(/\b(?:high|medium|low|urgent|critical)\s+priority\b/gi, ' ');
  value = value.replace(/\b(?:priority|category)\s*(?:is|:|=)?\s*[a-z]+\b/gi, ' ');
  value = value.replace(/\b(?:no|none)\s+priority\b/gi, ' ');
  value = value.replace(/[,\-–—:;]+/g, ' ');
  value = value.replace(/\s+/g, ' ').trim();
  value = value.replace(/\b(?:with|and|include|including|attach)\s*$/i, '').trim();

  const bareDate = findBareDate(value, now);
  if (bareDate) {
    value = value.replace(new RegExp(`\\b${escapeRegExp(bareDate.value)}\\b`, 'i'), ' ');
    value = value.replace(/\s+/g, ' ').trim();
  }
  if (timeMatch?.value) {
    value = value.replace(new RegExp(escapeRegExp(timeMatch.value), 'i'), ' ');
    value = value.replace(/\s+/g, ' ').trim();
  }
  value = value.replace(/^(?:for|to|about|on|at)\s+/i, '').trim();
  value = value.replace(/[.!?]+$/, '').trim();

  if (!value) return undefined;
  if (/^(?:today|tomorrow|yesterday|next\s+\w+|\d{4}-\d{2}-\d{2})$/i.test(value)) {
    return undefined;
  }
  return value.slice(0, 120);
}

/** True when the message clearly asks for a new personal task or flowchart task. */
export function detectTaskCreationIntent(raw: string): boolean {
  const text = normalize(raw);
  if (CANCEL_RE.test(text)) return false;
  return TASK_CREATE_RE.test(text);
}

/** True when the message mentions a flowchart/diagram/workflow. */
export function detectFlowchartIntent(raw: string): boolean {
  return FLOWCHART_RE.test(normalize(raw));
}

/** Merge a message into an existing draft, or start a new one. */
export function parseTaskDraft(
  raw: string,
  existing: TaskDraft | null = null,
  now: Date = new Date()
): TaskDraft {
  const text = normalize(raw);
  const draft: TaskDraft = existing
    ? { ...existing, flowchart: existing.flowchart }
    : { wantsFlowchart: false };

  // Chat asks the flowchart question only after the task itself is complete,
  // so a bare yes/no at that moment is about the flowchart, not a date.
  const answeringFlowchartQuestion = Boolean(
    existing &&
      existing.flowchartPrompted &&
      !existing.wantsFlowchart &&
      !existing.flowchart &&
      existing.title?.trim() &&
      existing.startDate &&
      existing.dueDate
  );

  if (SKIP_FLOWCHART_RE.test(text)) {
    draft.wantsFlowchart = false;
    draft.flowchart = undefined;
    draft.flowchartPrompted = true;
  }

  const flowchart = parseFlowchartSteps(text);
  if (flowchart) {
    draft.wantsFlowchart = true;
    draft.flowchart = flowchart;
    draft.flowchartPrompted = true;
  } else if (!existing && detectFlowchartIntent(text)) {
    draft.wantsFlowchart = true;
    draft.flowchartPrompted = true;
  }

  if (existing && SKIP_RE.test(text)) {
    if (answeringFlowchartQuestion || (draft.wantsFlowchart && !draft.flowchart)) {
      draft.wantsFlowchart = false;
      draft.flowchartPrompted = true;
    } else if (!draft.dueDate) {
      draft.dueDate = formatISO(now);
      if (!draft.startDate) draft.startDate = formatISO(now);
    } else if (!draft.startDate) {
      draft.startDate = formatISO(now);
    }
    return draft;
  }

  if (existing && ACCEPT_RE.test(text) && answeringFlowchartQuestion) {
    draft.wantsFlowchart = true;
    draft.flowchartPrompted = true;
    return draft;
  }

  const startDate = extractLabeledDate(text, START_LABELS, now);
  if (startDate) draft.startDate = startDate;

  const dueDate = extractLabeledDate(text, DUE_LABELS, now);
  if (dueDate) draft.dueDate = dueDate;

  const timeMatch = extractDueTime(text);
  if (timeMatch) draft.dueTime = timeMatch.time;

  const priority = extractPriority(text);
  if (priority === 'clear') draft.priority = undefined;
  else if (priority) draft.priority = priority;

  const category = extractCategory(text);
  if (category) draft.category = category;

  if (!draft.title) {
    const title = extractTitle(text, flowchart, now, timeMatch);
    if (title) draft.title = title;
  }

  const bare = findBareDate(text, now);
  if (bare) {
    // A date always fills the first open slot, but start date defaults to
    // today, so an unlabelled date is naturally the deadline.
    if (!draft.dueDate) draft.dueDate = bare.date;
    else if (!draft.startDate) draft.startDate = bare.date;
  }

  if (draft.dueDate && !draft.startDate) draft.startDate = formatISO(now);

  return draft;
}

/** Return the next conversational question, or null when the draft is complete. */
export function getTaskDraftPrompt(draft: TaskDraft, now: Date = new Date()): string | null {
  if (!draft.title?.trim()) return 'What should I call the task?';

  if (draft.wantsFlowchart && !draft.flowchart) {
    return 'Send the flowchart steps like `Research -> Design -> Launch`, or say `skip flowchart`.';
  }

  if (!draft.dueDate) {
    return 'What due date should I use? Try `YYYY-MM-DD`, `tomorrow`, or say `skip`.';
  }

  if (!draft.startDate) {
    return 'What start date should I use? Try `YYYY-MM-DD`, `today`, or say `skip`.';
  }

  const validationError = validateFlowchart(draft.flowchart);
  if (validationError) return `${validationError} Please send the steps again.`;

  const validation = validateChronologicalDraft(draft, now);
  if (!validation.valid) return validation.message;

  // Every task gets this question, even when no date or flowchart was requested.
  if (!draft.flowchartPrompted && !draft.wantsFlowchart) return FLOWCHART_PROMPT;

  return null;
}

function validateChronologicalDraft(
  draft: TaskDraft,
  now: Date
): { valid: boolean; message: string } {
  const result = validateChronologicalDeadline({
    startDate: draft.startDate,
    dueDate: draft.dueDate ?? '',
    dueTime: draft.dueTime,
    now,
  });
  if (result.isValid) return { valid: true, message: '' };

  if (result.startError) {
    return {
      valid: false,
      message: `${result.startError} Please give me a start date within the last 7 days.`,
    };
  }
  if (result.timeError) {
    return {
      valid: false,
      message: `${result.timeError} Please give me a later due time.`,
    };
  }
  if (result.dateError) {
    return {
      valid: false,
      message: `${result.generalError ?? result.dateError} Please give me a valid due date.`,
    };
  }
  return {
    valid: false,
    message: result.generalError ?? 'Those dates cannot be used together. Please try again.',
  };
}

/** Construct the final task from a complete draft. */
export function buildTaskFromDraft(draft: TaskDraft, now: Date = new Date()): SampleTask {
  const prompt = getTaskDraftPrompt(draft, now);
  if (prompt) throw new Error(prompt);
  return buildSampleTask({
    title: draft.title ?? '',
    description: draft.description,
    startDate: draft.startDate,
    dueDate: draft.dueDate ?? '',
    dueTime: draft.dueTime,
    priority: draft.priority,
    category: draft.category ?? 'personal',
    flowchart: draft.flowchart,
    now,
  });
}

const confirmationMessage = (task: SampleTask): string => {
  const parts = [`Done — I created the task "${task.title}"`];
  if (task.startDate) parts.push(`starting ${task.startDate}`);
  parts.push(`due ${task.dueDate}${task.dueTime ? ` at ${task.dueTime}` : ''}`);
  if (task.flowchart) parts.push(`with a ${task.flowchart.nodes.length}-step flowchart`);
  return `${parts.join(', ')}.`;
};

const continueDraft = (draft: TaskDraft, now: Date): TaskDraftResponse => {
  const prompt = getTaskDraftPrompt(draft, now);
  if (prompt) {
    if (prompt === FLOWCHART_PROMPT) draft.flowchartPrompted = true;
    return { handled: true, draft, reply: prompt, flowchart: draft.flowchart ?? undefined };
  }

  try {
    const task = buildTaskFromDraft(draft, now);
    return {
      handled: true,
      draft: null,
      reply: confirmationMessage(task),
      task,
      flowchart: task.flowchart,
    };
  } catch (error) {
    return {
      handled: true,
      draft,
      reply: `I couldn't create that task: ${
        error instanceof Error ? error.message : 'please try again'
      }`,
      flowchart: draft.flowchart ?? undefined,
    };
  }
};

/**
 * Main entry point used by Converse. Returns handled=false for ordinary chat
 * questions so they continue to the Python/answer pipeline.
 */
export function handleTaskDraftMessage(
  raw: string,
  existing: TaskDraft | null,
  now: Date = new Date()
): TaskDraftResponse {
  const text = String(raw || '').trim();
  if (!text) return { handled: false, draft: existing, reply: '' };

  if (!existing) {
    const wantsTask = detectTaskCreationIntent(text);
    const wantsFlowchart =
      detectFlowchartIntent(text) &&
      /\b(?:create|add|make|draw|build|generate)\b/i.test(text);
    if (!wantsTask && !wantsFlowchart) {
      return { handled: false, draft: existing, reply: '' };
    }
    return continueDraft(parseTaskDraft(text, null, now), now);
  }

  if (CANCEL_RE.test(text)) {
    return { handled: true, draft: null, reply: 'Okay, I cancelled that task draft.' };
  }

  return continueDraft(parseTaskDraft(text, existing, now), now);
}
