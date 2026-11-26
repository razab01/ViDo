import { describe, expect, it } from 'vitest';
import { getTodayDateString } from '../../utils/dateUtils';
import type { PythonPlanTask } from '../ml-lab/pythonApi';
import {
  bookingConfirmation,
  isDocumentBookingRequest,
  isGraphBookingRequest,
  localBookingTasks,
  plannedTaskToSampleTask,
} from './planBooking';

const dueDate = (): string => {
  const date = new Date();
  date.setDate(date.getDate() + 1);
  return getTodayDateString(date);
};

describe('planBooking', () => {
  it('turns a Graph RAG plan item into a canonical task', () => {
    const planned = {
      title: 'Draft spec',
      dueDate: dueDate(),
      dueTime: '09:30',
      priority: '',
      category: '',
      flowchartSteps: ['Draft', 'Review', 'Ship'],
    } as unknown as PythonPlanTask;

    const task = plannedTaskToSampleTask(planned);

    expect(task.title).toBe('Draft spec');
    expect(task.dueDate).toBe(planned.dueDate);
    expect(task.dueTime).toBe('09:30');
    expect(task.priority).toBeUndefined();
    expect(task.category).toBe('personal');
    expect(task.state).toBe('active');
    expect(task.flowchart).toEqual(
      expect.objectContaining({
        nodes: [
          expect.objectContaining({ label: 'Draft' }),
          expect.objectContaining({ label: 'Review' }),
          expect.objectContaining({ label: 'Ship' }),
        ],
      })
    );
    expect(bookingConfirmation([task])).toContain('All set — I booked "Draft spec"');
  });

  it.each([
    'book the task',
    'schedule task',
    'shedule the todo list',
    'create tasks according to this document',
  ])('recognises "%s" as document booking when a file is attached', (question) => {
    expect(isDocumentBookingRequest(question, true)).toBe(true);
  });

  it('leaves ordinary personal task creation with the local draft flow', () => {
    expect(isDocumentBookingRequest('create a task to review payroll tomorrow', true)).toBe(
      false
    );
    expect(isDocumentBookingRequest('book the task', false)).toBe(false);
  });

  it.each([
    'hey wanted to book todo task for tomorrow',
    'hey book a task for morning walk for tomorrow at 4am',
    'book a plan',
    'book a plan for tomorrow 4am as meet with US investors',
    'do task tomorrow',
    'bok a task for morning walk tomorrow',
  ])('recognises "%s" as a graph booking request', (question) => {
    expect(isGraphBookingRequest(question, true)).toBe(true);
    expect(isGraphBookingRequest(question, false)).toBe(true);
  });

  it('does not mistake ordinary creation for a graph booking', () => {
    expect(
      isGraphBookingRequest('create a task to review payroll tomorrow', true)
    ).toBe(false);
  });

  it('books a single described task offline', () => {
    const date = new Date();
    const tasks = localBookingTasks(
      'hey book a task for morning walk for tomorrow at 4am',
      date
    );
    expect(tasks).toHaveLength(1);
    expect(tasks?.[0]).toEqual(
      expect.objectContaining({
        title: 'morning walk',
        dueDate: dueDate(),
        dueTime: '04:00',
      })
    );
  });

  it('books a described plan offline without leaking connector words', () => {
    const date = new Date();
    const tasks = localBookingTasks(
      'book a plan for tomorrow 4am as meet with US investors',
      date
    );
    expect(tasks).toHaveLength(1);
    expect(tasks?.[0]).toEqual(
      expect.objectContaining({
        title: 'meet with US investors',
        dueDate: dueDate(),
        dueTime: '04:00',
      })
    );
  });
});
