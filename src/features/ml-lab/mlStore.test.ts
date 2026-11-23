import { beforeEach, describe, expect, it } from 'vitest';
import { profileData } from './engine';
import type { Row, TrainingResult } from './engine';
import {
  clearDatasetUploaded,
  clearSnapshot,
  formatTaskProbability,
  getActiveResult,
  hasDatasetUploaded,
  markDatasetUploaded,
  predictTask,
  readSnapshot,
  saveSnapshot,
  setActiveResult,
  toSerializableResult,
} from './mlStore';
import type { StoredMLLabState } from './mlStore';

const cols = ['task_text', 'priority', 'completed'];
const rows: Row[] = Array.from({ length: 12 }, (_, index) => ({
  task_text: `task ${index}`,
  priority: index % 2 ? 'High' : 'Low',
  completed: index % 3 ? 'yes' : 'no',
}));

const fakeResult: TrainingResult = {
  dataset: { cols, rows, target: 'completed' },
  profile: { ...profileData(cols, rows), target: 'completed', textCol: 'task_text' },
  target: 'completed',
  textCol: 'task_text',
  labels: ['no', 'yes'],
  features: ['task_text', 'priority'],
  splitNote: 'deterministic random split',
  capped: false,
  trainSize: 8,
  testSize: 4,
  tokenizerNote: 'test encoder',
  tokenizerWarning: '',
  tokenizer: { example: 'task 1', raw: [1], compact: [1], subwords: 'task', decoded: 'task' },
  history: { lt: [], le: [], at: [], ae: [] },
  metrics: [],
  best: { n: 'Fake best', acc: 0.75, P: 0.75, R: 0.75, F: 0.75, ms: 1 },
  artifact: { name: 'Fake best', predict: () => 1, proba: () => [0.2, 0.88] },
  suggestions: [],
  sample: { task_text: 'sample task', priority: 'Medium', completed: 'yes' },
  baseline: 0.5,
  binned: false,
};

const storedState = (
  target: string
): Omit<StoredMLLabState, 'version' | 'userId' | 'savedAt'> => ({
  upload: null,
  sheetName: '',
  dataset: { cols, rows, target },
  profile: profileData(cols, rows),
  target,
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

describe('ML Lab per-user store', () => {
  beforeEach(() => {
    localStorage.clear();
    clearSnapshot('user-a');
    clearSnapshot('user-b');
    clearDatasetUploaded('user-a');
    clearDatasetUploaded('user-b');
  });

  it('keeps saved datasets isolated by user id', () => {
    saveSnapshot('user-a', storedState('completed'));
    saveSnapshot('user-b', { ...storedState('priority'), target: 'priority' });

    expect(readSnapshot('user-a')?.target).toBe('completed');
    expect(readSnapshot('user-b')?.target).toBe('priority');
    expect(Object.keys(localStorage).filter((key) => key.startsWith('vido_ml_lab_v1_'))).toHaveLength(2);
  });

  it('serializes the result summary and predicts tasks only for the active user', () => {
    const serialized = toSerializableResult(fakeResult);
    expect(serialized.artifactName).toBe('Fake best');
    expect('artifact' in serialized).toBe(false);

    setActiveResult('user-a', fakeResult);
    markDatasetUploaded('user-a');
    expect(getActiveResult('user-a')).toBe(fakeResult);
    expect(predictTask('user-a', { title: 'Finish report', priority: 'High' })).toMatchObject({
      label: 'yes',
      probability: 0.88,
      model: 'Fake best',
      kind: 'confidence',
    });
    expect(predictTask('user-b', { title: 'Finish report' })).toBeNull();
  });

  it('does not predict from a saved model before data is loaded in this session', () => {
    setActiveResult('user-a', fakeResult);
    expect(hasDatasetUploaded('user-a')).toBe(false);
    expect(predictTask('user-a', { title: 'Finish report' })).toBeNull();
  });

  it('formats nearby confidences distinctly instead of rounding them all to 100%', () => {
    expect(formatTaskProbability(0.91)).toBe('91%');
    expect(formatTaskProbability(0.9876)).toBe('98.8%');
    expect(formatTaskProbability(0.9961)).toBe('99.6%');
    expect(formatTaskProbability(0.9999)).toBe('99.9%');
    expect(formatTaskProbability(1, 'test accuracy')).toBe('100%');
  });

  it('keeps distinct task estimates visible through predictTask', () => {
    const varying: TrainingResult = {
      ...fakeResult,
      artifact: {
        name: 'Logistic Regression',
        predict: () => 0,
        proba: (row) =>
          String(row[fakeResult.textCol]) === 'closing door' ? [0.9876, 0.0124] : [0.9961, 0.0039],
      },
    };
    setActiveResult('user-a', varying);
    markDatasetUploaded('user-a');

    const door = predictTask('user-a', { title: 'closing door' });
    const report = predictTask('user-a', { title: 'finish quarterly report' });

    expect(door && formatTaskProbability(door.probability, door.kind)).toBe('98.8%');
    expect(report && formatTaskProbability(report.probability, report.kind)).toBe('99.6%');
  });

  it('maps a task start date onto an added_date column in the saved schema', () => {
    const dateCols = ['task_text', 'added_date', 'deadline_date', 'completed'];
    const dateRows: Row[] = Array.from({ length: 12 }, (_, index) => ({
      task_text: `task ${index}`,
      added_date: `2026-11-${String(index + 1).padStart(2, '0')}`,
      deadline_date: `2026-11-${String(index + 2).padStart(2, '0')}`,
      completed: index % 2 ? 'yes' : 'no',
    }));
    const dateResult: TrainingResult = {
      ...fakeResult,
      dataset: { cols: dateCols, rows: dateRows, target: 'completed' },
      profile: { ...profileData(dateCols, dateRows), target: 'completed', textCol: 'task_text' },
      labels: ['no', 'yes'],
      sample: { task_text: 'sample task', added_date: '2024-01-01', deadline_date: '2024-01-02', completed: 'yes' },
      artifact: {
        name: 'Logistic Regression',
        predict: () => 1,
        proba: (row) => (String(row.added_date) === '2026-11-23' ? [0.2, 0.8] : [0.9, 0.1]),
      },
    };
    setActiveResult('user-a', dateResult);
    markDatasetUploaded('user-a');

    const prediction = predictTask('user-a', {
      title: 'closing door',
      startDate: '2026-11-23',
      dueDate: '2026-11-23',
    });

    expect(prediction?.probability).toBe(0.8);
  });
});
