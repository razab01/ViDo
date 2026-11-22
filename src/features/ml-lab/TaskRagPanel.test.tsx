import { beforeEach, describe, expect, it } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { SAMPLE_TASKS } from '../../data/sampleTasks';
import { profileData } from './engine';
import type { Row, TrainingResult } from './engine';
import { clearActiveResult, clearSnapshot, saveSnapshot, setActiveResult } from './mlStore';
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
  });

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

    render(<TaskRagPanel task={SAMPLE_TASKS[0]} userId="task-rag-user" />);

    expect(screen.getByText(/ask rag about this task/i)).toBeInTheDocument();
    expect(screen.getByText('tasks.xlsx')).toBeInTheDocument();
    expect(screen.getByText('Completion probability')).toBeInTheDocument();
    expect(screen.getByText('88%')).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('Ask RAG about this task'), {
      target: { value: 'When is this task due?' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }));

    expect(await screen.findByText(/17 Nov 2026 at 16:00/)).toBeInTheDocument();
    expect(screen.queryByText(/task[_-]?id/i)).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('Ask RAG about this task'), {
      target: { value: 'What is this task about?' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }));

    expect(await screen.findByText(/Ensure departmental forecasts/)).toBeInTheDocument();
    expect(screen.queryByText(/task[_-]?id/i)).not.toBeInTheDocument();
  });
});
