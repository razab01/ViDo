/**
 * Shared construction logic for personal tasks.
 *
 * Both the New Task modal and chat-created to-dos use this factory so dates,
 * checkpoints and attached flowcharts are saved in exactly the same shape.
 */

import { validateChronologicalDeadline } from '../utils/dateUtils';
import type { TaskFlowchart } from './flowchart';
import type { SampleTask, TaskCategory, TaskPriority } from './sampleTasks';

export interface SampleTaskInput {
  title: string;
  description?: string;
  startDate?: string;
  dueDate: string;
  dueTime?: string;
  priority?: TaskPriority;
  category: TaskCategory;
  flowchart?: TaskFlowchart | null;
  checkpointNote?: string;
  now?: Date;
  id?: string;
  checkpointId?: string;
}

/** Create the canonical active SampleTask after chronological validation. */
export function buildSampleTask(input: SampleTaskInput): SampleTask {
  const validation = validateChronologicalDeadline({
    startDate: input.startDate?.trim() || undefined,
    dueDate: input.dueDate,
    dueTime: input.dueTime,
    now: input.now,
  });

  if (!validation.isValid) {
    throw new Error(validation.generalError ?? 'Please provide valid task dates.');
  }

  const timestamp = validation.effectiveCreatedAt;
  const startDate = input.startDate?.trim() || undefined;
  const defaultNote = startDate
    ? `Task initialized in workspace (started on ${startDate}).`
    : 'Task initialized in workspace.';

  return {
    id: input.id ?? `task-${Date.now()}`,
    title: input.title.trim(),
    description: input.description?.trim() || undefined,
    startDate,
    dueDate: validation.effectiveDueDate,
    dueTime: validation.effectiveDueTime,
    priority: input.priority || undefined,
    state: 'active',
    category: input.category,
    createdAt: timestamp,
    flowchart: input.flowchart ?? undefined,
    checkpoints: [
      {
        id: input.checkpointId ?? `cp-${Date.now()}`,
        timestamp,
        note: input.checkpointNote ?? defaultNote,
      },
    ],
  };
}
