import { beforeEach, describe, expect, it } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { profileData } from './engine';
import type { Row } from './engine';
import { DashboardRagCard } from './DashboardRagCard';
import { clearSnapshot, saveSnapshot } from './mlStore';
import type { SerializableTrainingResult } from './mlStore';

const cols = ['task_text', 'priority', 'completed'];
const rows: Row[] = Array.from({ length: 12 }, (_, index) => ({
  task_text: `task ${index} for the team`,
  priority: index % 2 ? 'High' : 'Low',
  completed: index % 3 ? 'yes' : 'no',
}));

describe('DashboardRagCard', () => {
  beforeEach(() => {
    localStorage.clear();
    clearSnapshot('dash-rag-user');
  });

  it('surfaces the saved best model and answers a question from the dashboard', async () => {
    const profile = { ...profileData(cols, rows), target: 'completed', textCol: 'task_text' };
    const result: SerializableTrainingResult = {
      dataset: { cols, rows, target: 'completed' },
      profile,
      target: 'completed',
      textCol: 'task_text',
      labels: ['no', 'yes'],
      features: ['task_text', 'priority'],
      splitNote: 'random',
      capped: false,
      trainSize: 8,
      testSize: 4,
      tokenizerNote: 'test',
      tokenizerWarning: '',
      tokenizer: { example: 'task 1', raw: [], compact: [], subwords: '', decoded: '' },
      history: { lt: [], le: [], at: [], ae: [] },
      metrics: [],
      best: { n: 'Fake best', acc: 0.8, P: 0.8, R: 0.8, F: 0.8, ms: 1 },
      suggestions: [
        {
          id: 1,
          title: 'task 1 for the team',
          label: 'yes',
          conf: '80%',
          model: 'Fake best',
          target: 'completed',
          fields: 'priority=High',
        },
      ],
      sample: { task_text: 'task 1 for the team', priority: 'High', completed: 'yes' },
      baseline: 0.5,
      binned: false,
      artifactName: 'Fake best',
    };

    saveSnapshot('dash-rag-user', {
      upload: null,
      sheetName: '',
      dataset: { cols, rows, target: 'completed' },
      profile,
      target: 'completed',
      textCol: 'task_text',
      dateCol: '',
      tokenizer: 'builtin',
      split: 'auto',
      vocab: 2048,
      epochs: 2,
      options: null,
      result,
      predictionValues: {},
    });

    render(
      <MemoryRouter>
        <DashboardRagCard userId="dash-rag-user" basePath="/preview" />
      </MemoryRouter>
    );

    expect(screen.getByText(/model ready/i)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Ask your data'), {
      target: { value: 'Which tasks look likely to be completed?' },
    });
    fireEvent.click(screen.getByRole('button', { name: /ask rag/i }));

    expect(await screen.findByText(/^answer$/i)).toBeInTheDocument();
  });
});
