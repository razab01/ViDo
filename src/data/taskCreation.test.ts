import { describe, expect, it } from 'vitest';
import {
  buildTaskFromDraft,
  detectTaskCreationIntent,
  handleTaskDraftMessage,
  parseTaskDraft,
} from './taskCreation';

const NOW = new Date(2026, 10, 24, 9, 0, 0);

describe('conversational task creation', () => {
  it('detects creation requests but ignores dataset questions', () => {
    expect(detectTaskCreationIntent('ok create a task to pay bills tomorrow')).toBe(true);
    expect(detectTaskCreationIntent('please add a todo for the report')).toBe(true);
    expect(detectTaskCreationIntent('which tasks are completed?')).toBe(false);
  });

  it('extracts a complete task draft, dates, priority, category and flowchart', () => {
    const draft = parseTaskDraft(
      'create a high priority work task to finish report from 2026-11-23 due 2026-11-24 at 5pm with flowchart Research -> Draft -> Review',
      null,
      NOW
    );

    expect(draft.title).toBe('finish report');
    expect(draft.startDate).toBe('2026-11-23');
    expect(draft.dueDate).toBe('2026-11-24');
    expect(draft.dueTime).toBe('17:00');
    expect(draft.priority).toBe('high');
    expect(draft.category).toBe('work');
    expect(draft.flowchart?.nodes.map((node) => node.label)).toEqual([
      'Research',
      'Draft',
      'Review',
    ]);

    const task = buildTaskFromDraft(draft, NOW);
    expect(task.title).toBe('finish report');
    expect(task.state).toBe('active');
    expect(task.flowchart?.nodes).toHaveLength(3);
  });

  it('understands relative dates and morning times, then always offers a flowchart', () => {
    const first = handleTaskDraftMessage(
      'hey create a task for tomorrow morning 4 am for walk.',
      null,
      NOW
    );

    expect(first.handled).toBe(true);
    expect(first.reply).toMatch(/would you like a flowchart/i);
    expect(first.task).toBeUndefined();
    expect(first.draft).toEqual(
      expect.objectContaining({
        title: 'walk',
        startDate: '2026-11-24',
        dueDate: '2026-11-25',
        dueTime: '04:00',
        flowchartPrompted: true,
      })
    );

    const accepted = handleTaskDraftMessage('yes', first.draft, NOW);
    expect(accepted.reply).toMatch(/flowchart steps/i);
    expect(accepted.task).toBeUndefined();

    const declinedSteps = handleTaskDraftMessage('skip flowchart', accepted.draft, NOW);
    expect(declinedSteps.reply).toMatch(/done/i);
    expect(declinedSteps.task).toEqual(
      expect.objectContaining({
        title: 'walk',
        startDate: '2026-11-24',
        dueDate: '2026-11-25',
        dueTime: '04:00',
      })
    );
  });

  it('creates a task without a flowchart when the user declines the prompt', () => {
    const first = handleTaskDraftMessage('create a task to review payroll tomorrow', null, NOW);
    expect(first.reply).toMatch(/would you like a flowchart/i);
    expect(first.draft?.startDate).toBe('2026-11-24');
    expect(first.draft?.dueDate).toBe('2026-11-25');

    const second = handleTaskDraftMessage('no', first.draft, NOW);
    expect(second.reply).toMatch(/done/i);
    expect(second.draft).toBeNull();
    expect(second.task).toEqual(
      expect.objectContaining({
        title: 'review payroll',
        startDate: '2026-11-24',
        dueDate: '2026-11-25',
      })
    );
  });

  it('collects flowchart steps interactively and attaches them to the task', () => {
    const first = handleTaskDraftMessage('create a flowchart for Launch plan', null, NOW);
    expect(first.reply).toMatch(/flowchart steps/i);

    const second = handleTaskDraftMessage('Research -> Design -> Build', first.draft, NOW);
    expect(second.reply).toMatch(/due date/i);
    expect(second.flowchart?.nodes).toHaveLength(3);

    const third = handleTaskDraftMessage('tomorrow', second.draft, NOW);
    expect(third.reply).toMatch(/3-step flowchart/i);
    expect(third.task?.flowchart?.nodes.map((node) => node.label)).toEqual([
      'Research',
      'Design',
      'Build',
    ]);
  });

  it('cancels an in-progress draft', () => {
    const first = handleTaskDraftMessage('add a task to call mom', null, NOW);
    const cancelled = handleTaskDraftMessage('cancel', first.draft, NOW);

    expect(cancelled.handled).toBe(true);
    expect(cancelled.draft).toBeNull();
    expect(cancelled.reply).toMatch(/cancelled/i);
  });

  it('asks again when a date combination is invalid', () => {
    const attempt = handleTaskDraftMessage(
      'create a task to test from 2026-11-24 due 2026-11-20',
      null,
      NOW
    );
    expect(attempt.reply).toMatch(/deadline|due date/i);
    expect(attempt.task).toBeUndefined();
  });
});
