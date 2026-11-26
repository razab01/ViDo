/**
 * Turns Graph RAG's structured plan items into normal workspace tasks.
 *
 * Graph RAG understands the document and plans the booking; this helper only
 * constructs the canonical SampleTask so the existing Converse callback can
 * persist it exactly like a manually created task.
 */

import { parseFlowchartSteps } from '../../data/flowchart';
import type { SampleTask } from '../../data/sampleTasks';
import { buildSampleTask } from '../../data/taskFactory';
import { parseTaskDraft } from '../../data/taskCreation';
import type { PythonPlanTask } from '../ml-lab/pythonApi';

const optionalText = (value?: string): string | undefined => {
  const text = String(value || '').trim();
  return text || undefined;
};

const DOC_BOOKING_VERB_RE =
  /\b(?:book|schedul\w*|shedul\w*|scedul\w*|create|add|save|turn)\b/i;
const DEFINITE_PLAN_OBJECT_RE =
  /\b(?:the|this|that|my|our|all|these|attached|uploaded)\s+(?:tasks?|to-?do|todos?|plans?|agenda|list|handbook)\b/i;
const BARE_BOOKING_RE =
  /\b(?:book|schedul\w*|shedul\w*|scedul\w*)\s+(?:tasks?|to-?do|todos?|plans?|agenda|list|it|this|these|them)\b/i;
const DOCUMENT_REFERENCE_RE =
  /\b(?:from|according\s+to|based\s+on|in|of)\s+(?:it|this|that|the\s+(?:document|file|handbook|plan|list)|attached|uploaded)\b/i;
const ORDINARY_SINGLE_TASK_RE =
  /\b(?:create|add|make)\s+(?:a|an|new)\s+(?:task|to-?do|todo|reminder)\b/i;
const ROUTING_PREFIX_RE =
  /^(?:(?:hey|hi|hello|yo|howdy|ok|okay|so|well|please|pls|plz|thanks|thank\s+you)\b[\s,.:-]*)*(?:(?:i\s+)?(?:wanna|want|wanted|would\s+like|need)\s+to\s+)?/i;
const SINGLE_BOOKING_COMMAND_RE =
  /^(?:book|schedul\w*|shedul\w*|scedul\w*|bok|do|set\s*up)\s+(?:(?:a|an|one|new|my|the|this|that)\s+)*(?:single\s+)?(?:to-?do(?:\s+task)?|tasks?|todos?|reminders?|plans?)\b[\s,:;-]*/i;
const SINGLE_BOOKING_RE =
  /^(?:book|schedul\w*|shedul\w*|scedul\w*|bok|do|set\s*up)\s+(?:(?:a|an|one|new|my|the|this|that)\s+)*(?:single\s+)?(?:to-?do(?:\s+task)?|tasks?|todos?|reminders?|plans?)\b/i;
const LEADING_CONNECTOR_RE = /^\s*(?:for|about|as|at|on|to(?!-))\b[\s,:;-]*/i;
const ROUTING_TYPOS: Array<[RegExp, string]> = [
  [/\b(?:scedul\w*|schedul|schedue|schedual|shedule|shedul|schdule|scheule)\b/gi, 'schedule'],
  [/\b(?:bok|buok|bookk|boook)\b/gi, 'book'],
  [/\b(?:taskt|taskk|taks)\b/gi, 'task'],
  [/\b(?:todot|toodos|todoos)\b/gi, 'todo'],
  [/\b(?:tomm?orow|tomorow|2moro|2morow)\b/gi, 'tomorrow'],
];

const normaliseRouting = (raw: string): string => {
  let text = String(raw || '').trim();
  for (const [pattern, replacement] of ROUTING_TYPOS) text = text.replace(pattern, replacement);
  return text.replace(ROUTING_PREFIX_RE, '').trim();
};

/**
 * True when a short request such as "book the task" or "schedule task" points
 * at the attached plan. Converse's personal-task draft must step aside here so
 * Graph RAG can own the booking, including its clarification state.
 */
export function isDocumentBookingRequest(
  question: string,
  hasDocument: boolean
): boolean {
  if (!hasDocument) return false;
  const text = normaliseRouting(question);
  if (ORDINARY_SINGLE_TASK_RE.test(text) && !DOCUMENT_REFERENCE_RE.test(text)) {
    return false;
  }
  return (
    DOC_BOOKING_VERB_RE.test(text) &&
    (DEFINITE_PLAN_OBJECT_RE.test(text) ||
      BARE_BOOKING_RE.test(text) ||
      DOCUMENT_REFERENCE_RE.test(text))
  );
}

/**
 * True for both "book the tasks in my attachment" and a fresh request such as
 * "book a task for morning walk tomorrow at 4am". These belong to Graph RAG
 * booking, never to the personal-task draft or a document excerpt.
 */
export function isGraphBookingRequest(
  question: string,
  hasDocument: boolean
): boolean {
  const text = normaliseRouting(question);
  return SINGLE_BOOKING_RE.test(text) || isDocumentBookingRequest(text, hasDocument);
}

/**
 * Offline twin for a single described task. The Python route remains preferred;
 * this keeps "book a task for X tomorrow" usable when the service is down.
 */
export function localBookingTasks(
  question: string,
  now: Date = new Date()
): PythonPlanTask[] | null {
  const text = normaliseRouting(question);
  const command = SINGLE_BOOKING_COMMAND_RE.exec(text);
  if (!command) return null;

  const detail = text.slice(command[0].length).replace(LEADING_CONNECTOR_RE, '').trim();
  if (!detail) return null;
  const draft = parseTaskDraft(detail, null, now);
  let title = draft.title?.trim() ?? '';
  while (LEADING_CONNECTOR_RE.test(title)) {
    title = title.replace(LEADING_CONNECTOR_RE, '').trim();
  }
  while (/\s+(?:for|about|as|on|at|to|from|in|of)\s*$/i.test(title)) {
    title = title.replace(/\s+(?:for|about|as|on|at|to|from|in|of)\s*$/i, '').trim();
  }
  if (!title || !draft.dueDate) return null;

  return [
    {
      title,
      description: draft.description || '',
      startDate: draft.startDate || '',
      dueDate: draft.dueDate,
      dueTime: draft.dueTime || '',
      priority: draft.priority || '',
      category: draft.category || '',
      flowchartSteps: draft.flowchart?.nodes.map((node) => node.label) ?? [],
    } as unknown as PythonPlanTask,
  ];
}

export function plannedTaskToSampleTask(
  task: PythonPlanTask,
  now: Date = new Date()
): SampleTask {
  const steps = (task.flowchartSteps ?? [])
    .map((step) => String(step || '').trim())
    .filter(Boolean)
    .join(' -> ');
  const priority = optionalText(task.priority);
  const category = optionalText(task.category);

  return buildSampleTask({
    title: task.title || 'Untitled plan',
    description: optionalText(task.description),
    startDate: optionalText(task.startDate),
    dueDate: optionalText(task.dueDate) || '',
    dueTime: optionalText(task.dueTime),
    priority:
      priority === 'high' || priority === 'medium' || priority === 'low'
        ? priority
        : undefined,
    category:
      category === 'work' || category === 'home' || category === 'personal'
        ? category
        : 'personal',
    flowchart: steps ? parseFlowchartSteps(steps) : undefined,
    now,
  });
}

const friendlyTime = (value?: string): string => {
  const match = /^(\d{1,2}):(\d{2})$/.exec(String(value || ''));
  if (!match) return String(value || '');
  const hour = Number(match[1]);
  const suffix = hour >= 12 ? 'pm' : 'am';
  return `${hour % 12 || 12}:${match[2]} ${suffix}`;
};

const bookedWhen = (task: SampleTask): string =>
  [task.dueDate, task.dueTime ? friendlyTime(task.dueTime) : ''].filter(Boolean).join(' at ');

export function bookingConfirmation(tasks: SampleTask[]): string {
  if (!tasks.length) return 'There was nothing to book.';
  if (tasks.length === 1) {
    const task = tasks[0];
    const when = bookedWhen(task);
    return `All set — I booked "${task.title}"${when ? ` for ${when}` : ''}.`;
  }
  const titles = tasks.map((task) => `"${task.title}"`).join(', ');
  return `All set — I booked ${tasks.length} tasks: ${titles}.`;
}
