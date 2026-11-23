import { beforeEach, describe, expect, it } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { profileData } from './engine';
import type { Row } from './engine';
import { MLLabPage } from './MLLabPage';
import {
  clearDatasetUploaded,
  clearSnapshot,
  saveSnapshot,
} from './mlStore';
import type { SerializableTrainingResult } from './mlStore';

const setOfflineTrainingOptions = () => {
  fireEvent.click(screen.getByRole('button', { name: /advanced options/i }));
  fireEvent.change(screen.getByLabelText(/gpt tokenizer/i), { target: { value: 'builtin' } });
  fireEvent.click(screen.getByRole('button', { name: /hide advanced options/i }));
};

describe('MLLabPage', () => {
  beforeEach(() => {
    clearDatasetUploaded('mllab-train-user');
    clearDatasetUploaded('mllab-empty-rag-user');
    clearDatasetUploaded('mllab-hide-pct-user');
  });

  it('trains automatically after upload and keeps technical details out of the simple view', async () => {
    render(<MLLabPage userId="mllab-train-user" />);

    expect(screen.getByText(/upload a file to get started/i)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /train 10 deep models/i })).not.toBeInTheDocument();

    setOfflineTrainingOptions();
    fireEvent.click(screen.getByRole('button', { name: /load demo data/i }));

    expect(await screen.findByText(/learning from your file/i)).toBeInTheDocument();
    expect(await screen.findByText(/ready to answer questions/i, {}, { timeout: 30000 })).toBeInTheDocument();
    expect(screen.getByText(/learned from 300 rows/i)).toBeInTheDocument();
    expect(screen.queryByText(/tokenizer verification/i)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /advanced options/i }));
    expect(screen.getByText(/tokenizer verification/i)).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /suggestions/i })).toBeInTheDocument();
    expect(screen.getByText(/always uses graph rag/i)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Graph' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Agentic' })).not.toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText(/ask about your data or any topic/i), {
      target: { value: 'Which task_text records look likely to be completed?' },
    });
    fireEvent.click(screen.getByRole('button', { name: /find answer/i }));

    expect(screen.getByText(/finding the answer/i)).toBeInTheDocument();
    expect(await screen.findByText(/^answer$/i)).toBeInTheDocument();
  }, 60000);

  it('shows a friendly hint instead of an error badge for an empty question', async () => {
    localStorage.clear();
    render(<MLLabPage userId="mllab-empty-rag-user" />);

    setOfflineTrainingOptions();
    fireEvent.click(screen.getByRole('button', { name: /load demo data/i }));
    expect(await screen.findByText(/ready to answer questions/i, {}, { timeout: 30000 })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /find answer/i }));

    expect(screen.getByText(/type a question or pick an example/i)).toBeInTheDocument();
    expect(screen.queryByText('NONE')).not.toBeInTheDocument();
  }, 60000);

  it('hides every saved-model percentage until a dataset is loaded in this session', async () => {
    const userId = 'mllab-hide-pct-user';
    localStorage.clear();
    clearSnapshot(userId);
    clearDatasetUploaded(userId);

    const cols = ['task_text', 'completed'];
    const rows: Row[] = Array.from({ length: 8 }, (_, index) => ({
      task_text: `task ${index} for the team`,
      completed: index % 2 ? 'no' : 'yes',
    }));
    const profile = { ...profileData(cols, rows), target: 'completed', textCol: 'task_text' };
    const metrics = { n: 'Saved best', acc: 0.8, P: 0.8, R: 0.8, F: 0.8, ms: 1 };
    const result: SerializableTrainingResult = {
      dataset: { cols, rows, target: 'completed' },
      profile,
      target: 'completed',
      textCol: 'task_text',
      labels: ['no', 'yes'],
      features: ['task_text'],
      splitNote: 'random',
      capped: false,
      trainSize: 6,
      testSize: 2,
      tokenizerNote: 'test',
      tokenizerWarning: '',
      tokenizer: { example: 'task 0', raw: [], compact: [], subwords: '', decoded: '' },
      history: { lt: [], le: [], at: [], ae: [] },
      metrics: [metrics],
      best: metrics,
      suggestions: [
        {
          id: 1,
          title: 'task 0 for the team',
          label: 'yes',
          conf: '80.0%',
          model: 'Saved best',
          target: 'completed',
          fields: 'task_text=task 0',
        },
      ],
      sample: { task_text: 'task 0 for the team', completed: 'yes' },
      baseline: 0.5,
      binned: false,
      artifactName: 'Saved best',
    };
    saveSnapshot(userId, {
      upload: null,
      sheetName: '',
      dataset: result.dataset,
      profile,
      target: 'completed',
      textCol: 'task_text',
      dateCol: '',
      tokenizer: 'builtin',
      split: 'auto',
      vocab: 512,
      epochs: 2,
      options: null,
      result,
      predictionValues: {},
    });

    render(<MLLabPage userId={userId} />);
    expect(await screen.findByText(/ready to answer questions/i)).toBeInTheDocument();
    expect(screen.queryByText(/accuracy on test rows/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/80\.0%/)).not.toBeInTheDocument();
    expect(screen.getAllByText('—').length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole('button', { name: /advanced options/i }));
    expect(screen.queryByText(/80\.0% chance/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/training accuracy chart/i)).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/gpt tokenizer/i), { target: { value: 'builtin' } });
    fireEvent.click(screen.getByRole('button', { name: /load demo data/i }));

    expect(await screen.findByText(/accuracy on test rows/i, {}, { timeout: 30000 })).toBeInTheDocument();
    expect(screen.getAllByText(/\d+\.\d+%/).length).toBeGreaterThan(0);
  }, 60000);
});
