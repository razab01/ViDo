import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { SAMPLE_TASKS } from '../../data/sampleTasks';
import type { SampleTask } from '../../data/sampleTasks';
import { profileData } from './engine';
import type { Row, TrainingResult } from './engine';
import {
  clearActiveResult,
  clearDatasetUploaded,
  clearSnapshot,
  markDatasetUploaded,
  saveSnapshot,
  setActiveResult,
} from './mlStore';
import { TaskRagPanel } from './TaskRagPanel';

const cols = ['task_text', 'priority', 'completed'];
const rows: Row[] = Array.from({ length: 12 }, (_, index) => ({
  task_text: `task ${index} for the team`,
  priority: index % 2 ? 'High' : 'Low',
  completed: index % 3 ? 'yes' : 'no',
}));

const profile = { ...profileData(cols, rows), target: 'completed', textCol: 'task_text' };
const result: TrainingResult = {
  dataset: { cols, rows, target: 'completed' },
  profile,
  target: 'completed',
  textCol: 'task_text',
  labels: ['no', 'yes'],
  features: ['task_text', 'priority'],
  splitNote: 'test split',
  capped: false,
  trainSize: 8,
  testSize: 4,
  tokenizerNote: 'test',
  tokenizerWarning: '',
  tokenizer: { example: 'task 0', raw: [1], compact: [1], subwords: 'task', decoded: 'task' },
  history: { lt: [], le: [], at: [], ae: [] },
  metrics: [],
  best: { n: 'Fake best', acc: 0.8, P: 0.8, R: 0.8, F: 0.8, ms: 1 },
  artifact: { name: 'Fake best', predict: () => 1, proba: () => [0.12, 0.88] },
  suggestions: [],
  sample: { task_text: 'sample task', priority: 'High', completed: 'yes' },
  baseline: 0.5,
  binned: false,
};

describe('TaskRagPanel', () => {
  beforeEach(() => {
    localStorage.clear();
    clearSnapshot('task-rag-user');
    clearActiveResult('task-rag-user');
    clearDatasetUploaded('task-rag-user');
    vi.stubGlobal('fetch', vi.fn(async () => { throw new Error('offline'); }));
  });

  afterEach(() => vi.unstubAllGlobals());

  it('shows the auto-detected source and answers questions about the selected task', async () => {
    const dataset = result.dataset;
    saveSnapshot('task-rag-user', {
      upload: {
        name: 'tasks.xlsx',
        names: ['tasks.xlsx'],
        sheets: { Tasks: { cols, rows } },
        sheet: 'Tasks',
        data: { dataset, profile, valid: true, message: '' },
      },
      sheetName: 'Tasks',
      dataset,
      profile,
      target: 'completed',
      textCol: 'task_text',
      dateCol: '',
      tokenizer: 'builtin',
      split: 'auto',
      vocab: 2048,
      epochs: 2,
      options: null,
      result: null,
      predictionValues: {},
    });
    setActiveResult('task-rag-user', result);
    markDatasetUploaded('task-rag-user');

    render(<TaskRagPanel task={SAMPLE_TASKS[0]} userId="task-rag-user" />);

    expect(screen.getByText(/ask about this task/i)).toBeInTheDocument();
    expect(screen.getByText('tasks.xlsx')).toBeInTheDocument();
    expect(screen.getByText('Completion probability')).toBeInTheDocument();
    expect(screen.getByText('88%')).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('Ask about this task'), {
      target: { value: 'When is this task due?' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }));

    expect(await screen.findByText(/17 Nov 2026 at 16:00/)).toBeInTheDocument();
    expect(screen.queryByText(/task[_-]?id/i)).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('Ask about this task'), {
      target: { value: 'What is this task about?' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }));

    expect(await screen.findByText(/ensure departmental forecasts/i)).toBeInTheDocument();
    expect(screen.queryByText(/task[_-]?id/i)).not.toBeInTheDocument();
  });

  it('hides the saved-model probability until data is loaded in this session', () => {
    setActiveResult('task-rag-user', result);

    render(<TaskRagPanel task={SAMPLE_TASKS[0]} userId="task-rag-user" />);

    expect(screen.queryByText('Completion probability')).not.toBeInTheDocument();
    expect(screen.queryByText('88%')).not.toBeInTheDocument();
  });

  it('explains what a Boating task is instead of echoing its title and category', async () => {
    const boating: SampleTask = {
      id: 'boating-task',
      title: 'Boating',
      description: 'Plan a relaxed day out.',
      priority: 'medium',
      state: 'active',
      category: 'personal',
      createdAt: '2026-11-20T09:00:00Z',
      dueDate: '2026-11-25',
    };

    render(<TaskRagPanel task={boating} userId="task-rag-user" />);

    fireEvent.change(screen.getByLabelText('Ask about this task'), {
      target: { value: 'What is this task about?' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }));

    expect(await screen.findByText(/Boating is spending time on the water/i)).toBeInTheDocument();
    expect(screen.queryByText('This task is “Boating” in the personal category.')).not.toBeInTheDocument();
  });

  it('explains a financial report from every word instead of a status-report keyword', async () => {
    const report: SampleTask = {
      id: 'report-task',
      title: 'financial report for this quarter',
      priority: 'medium',
      state: 'active',
      category: 'work',
      createdAt: '2026-11-20T09:00:00Z',
      dueDate: '2026-11-23',
      dueTime: '23:59',
    };

    render(<TaskRagPanel task={report} userId="task-rag-user" />);

    fireEvent.change(screen.getByLabelText('Ask about this task'), {
      target: { value: 'What is this task about?' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }));

    expect(await screen.findByText(/clear summary of what happened/i)).toBeInTheDocument();
    expect(screen.getByText(/money, budgets and accounts/i)).toBeInTheDocument();
    expect(screen.getByText(/three-month period/i)).toBeInTheDocument();
    expect(screen.queryByText(/a good status report/i)).not.toBeInTheDocument();
  });

  it('checks individual words on Wikipedia when a task title has no exact page', async () => {
    const calculus: SampleTask = {
      id: 'calculus-task',
      title: 'Schedule for calculus class',
      priority: 'medium',
      state: 'active',
      category: 'personal',
      createdAt: '2026-11-20T09:00:00Z',
      dueDate: '2026-11-23',
      dueTime: '23:59',
    };

    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('list=search')) {
        return {
          ok: true,
          json: async () => ({ query: { search: [{ title: 'Calculus' }] } }),
        } as unknown as Response;
      }
      if (url.includes('titles=Calculus')) {
        return {
          ok: true,
          json: async () => ({
            query: {
              pages: {
                '5176': {
                  title: 'Calculus',
                  extract:
                    'Calculus is the branch of mathematics that studies continuous change, and is the principal precursor of modern mathematical analysis.',
                },
              },
            },
          }),
        } as unknown as Response;
      }
      return {
        ok: true,
        json: async () => ({ query: { pages: { '-1': { extract: '' } } } }),
      } as unknown as Response;
    });
    vi.stubGlobal('fetch', fetchMock);

    render(<TaskRagPanel task={calculus} userId="task-rag-user" />);

    fireEvent.change(screen.getByLabelText('Ask about this task'), {
      target: { value: 'What is this task about?' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }));

    expect(await screen.findByText(/the branch of mathematics that studies continuous change/i)).toBeInTheDocument();
    expect(await screen.findByText(/principal precursor of modern mathematical analysis/i)).toBeInTheDocument();
    expect(fetchMock.mock.calls.some((call) => String(call[0]).includes('srsearch=calculus'))).toBe(true);
  });
});
