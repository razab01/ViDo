import { beforeAll, describe, expect, it } from 'vitest';
import {
  formatConfidence,
  normalizeTable,
  parseCSV,
  predictWith,
  prepareDataset,
  profileData,
  rankModels,
  ragDocs,
  runRAG,
  syncPredictionRow,
  trainAll,
  USER_RAG_MODE,
} from './engine';
import type { RAGDoc, Row, TrainingOptions, TrainingResult } from './engine';

const options = (overrides: Partial<TrainingOptions> = {}): TrainingOptions => ({
  target: 'completed',
  textCol: 'task_text',
  tokenizer: 'builtin',
  split: 'random',
  vocab: 512,
  epochs: 2,
  bpeEmb: 8,
  bpeHidden: 12,
  bpeEpochs: 2,
  ...overrides,
});

function trainingRows(count: number): Row[] {
  return Array.from({ length: count }, (_, i) => {
    const positive = i % 2 === 0;
    return {
      priority: positive ? 'High' : 'Low',
      scope: i % 3 === 0 ? 'team' : 'private',
      est_hours: String(1 + (i % 7)),
      task_text: positive
        ? `finished delivery ${i} smoothly ahead of schedule`
        : `blocked review ${i} pending overdue needs review`,
      added_date: `2026-01-${String((i % 28) + 1).padStart(2, '0')}`,
      completed: positive ? 'yes' : 'no',
    };
  });
}

describe('dynamic ingestion and schema inference', () => {
  it('parses quoted delimited files with a detected separator', () => {
    const table = normalizeTable(parseCSV('task\tdate\tcompleted\n"Ship, release"\t46000\tyes\nFix bug\t46001\tno\n'));
    expect(table.cols).toEqual(['task', 'date', 'completed']);
    expect(table.rows[0].task).toBe('Ship, release');
    expect(table.rows[0].date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });

  it('infers target, text, date and leakage guard from a derived target', () => {
    const rows: Row[] = Array.from({ length: 12 }, (_, i) => ({
      task: i % 2 ? 'Review draft' : 'Prepare release',
      added_date: `2026-02-${String(i + 1).padStart(2, '0')}`,
      completed_date: i % 2 ? '' : `2026-02-${String(i + 2).padStart(2, '0')}`,
      owner: `Person ${i}`,
    }));
    const prepared = prepareDataset(['task', 'added_date', 'completed_date', 'owner'], rows);
    expect(prepared.valid).toBe(true);
    expect(prepared.dataset.target).toBe('completed');
    expect(prepared.profile.targetSource).toBe('completed_date');
    expect(prepared.profile.textCol).toBe('task');
    expect(prepared.profile.primaryDate).toBe('added_date');
    expect(prepared.dataset.cols).toContain('completed');
    expect(prepared.dataset.cols).toContain('added_date_weekday');
    expect(prepared.dataset.cols).toContain('text_length');
  });

  it('respects an explicitly selected target column', () => {
    const rows = trainingRows(20);
    const prepared = prepareDataset(Object.keys(rows[0]), rows, { target: 'priority' });
    expect(prepared.dataset.target).toBe('priority');
    expect(prepared.profile.target).toBe('priority');
  });
});

describe('confidence formatting', () => {
  it('keeps saturated probabilities just below 100%', () => {
    expect(formatConfidence(0.88)).toBe('88%');
    expect(formatConfidence(0.999999)).toBe('99.9%');
  });
});

describe('model ranking', () => {
  it('prefers the LSTM when the top models have identical scores', () => {
    const tied = [
      { n: 'Logistic Regression', acc: 0.9, F: 0.9 },
      { n: 'Decision Tree', acc: 0.9, F: 0.9 },
      { n: 'LSTM · GPT token IDs', acc: 0.9, F: 0.9 },
    ];
    expect(rankModels(tied)[0].n).toBe('LSTM · GPT token IDs');
  });

  it('still lets a strictly better non-LSTM model win', () => {
    const models = [
      { n: 'LSTM · GPT token IDs', acc: 0.9, F: 0.9 },
      { n: 'Logistic Regression', acc: 0.94, F: 0.94 },
    ];
    expect(rankModels(models)[0].n).toBe('Logistic Regression');
  });
});

describe('unified training, prediction and RAG', () => {
  let trained: TrainingResult;

  beforeAll(async () => {
    const rows = trainingRows(80);
    const prepared = prepareDataset(Object.keys(rows[0]), rows);
    trained = await trainAll(prepared.dataset, prepared.profile, options(), () => undefined, () => undefined);
  }, 30000);

  it('trains ten models and keeps the best artifact usable for prediction', () => {
    expect(trained.metrics).toHaveLength(10);
    expect(trained.best.n.length).toBeGreaterThan(0);
    expect(trained.suggestions.length).toBeGreaterThan(0);
    expect(trained.suggestions.length).toBeLessThanOrEqual(8);
    expect(trained.tokenizer.compact.length).toBeGreaterThan(0);
    expect(trained.history.ae.length).toBe(2);

    const prediction = predictWith(trained, trained.sample);
    expect(trained.labels).toContain(prediction.label);
    expect(prediction.model).toBe(trained.best.n);

    const text = 'delivered final report 42';
    const synced = syncPredictionRow(trained, { ...trained.sample, [trained.textCol]: text });
    expect(synced.text_length).toBe(String(text.length));
    expect(synced.word_count).toBe('4');
  });

  it('auto-bins a numeric target into classes', async () => {
    const rows: Row[] = Array.from({ length: 60 }, (_, i) => ({
      product: i % 2 ? `Refunded order ${i}` : `Delivered order ${i}`,
      category: i % 3 ? 'home' : 'office',
      revenue: String(100 + i * 17),
    }));
    const prepared = prepareDataset(Object.keys(rows[0]), rows);
    const result = await trainAll(
      prepared.dataset,
      prepared.profile,
      options({ target: 'revenue', textCol: 'product', bpeHidden: 10, bpeEmb: 6 }),
      () => undefined,
      () => undefined
    );
    expect(result.binned).toBe(true);
    expect(result.target).toBe('revenue (bins)');
    expect(result.labels.length).toBeGreaterThanOrEqual(3);
  }, 30000);

  it('uses schema-aware documents and honors manual RAG overrides', () => {
    const docs = ragDocs(trained.dataset, trained.profile, trained.suggestions);
    expect(docs.length).toBe(trained.dataset.rows.length + trained.suggestions.length);
    expect(docs.some((doc) => doc.suggestion)).toBe(true);

    for (const mode of ['rag', 'agentic', 'graph'] as const) {
      const response = runRAG('finished delivery', mode, docs, trained.profile, trained.suggestions);
      expect(response.mode).toBe(mode);
      expect(response.answer.length).toBeGreaterThan(0);
    }

    const empty = runRAG('', 'auto', docs, trained.profile, trained.suggestions);
    expect(empty.mode).toBe('none');
  });

  it('uses one fixed agentic strategy for user-facing answers', () => {
    expect(USER_RAG_MODE).toBe('agentic');
    const docs = ragDocs(trained.dataset, trained.profile, trained.suggestions);
    const response = runRAG('finished delivery', USER_RAG_MODE, docs, trained.profile, trained.suggestions);
    expect(response.mode).toBe('agentic');
    expect(response.plan.length).toBeGreaterThan(0);
  });

  it('extracts the sentence that directly answers a due-date question', () => {
    const doc: RAGDoc = {
      id: 'task-due-test',
      title: 'Review budget',
      text:
        'Task title: Review budget. Description: Check forecasts. ' +
        'Due date: 2026-11-17. Due time: 16:00.',
      tags: '',
      priority: '',
      assignee: '',
      edges: [],
    };
    const response = runRAG('When is this task due?', 'rag', [doc], trained.profile, []);
    expect(response.answer).toContain('2026-11-17');
  });

  it('keeps identifier columns out of RAG document text', () => {
    const cols = ['task_id', 'task_text', 'due_date', 'completed'];
    const rows = [
      { task_id: 'T09', task_text: 'Morning walk', due_date: '2026-11-23', completed: 'no' },
      { task_id: 'T03', task_text: 'Review budget', due_date: '2026-11-24', completed: 'yes' },
    ];
    const profile = { ...profileData(cols, rows), target: 'completed', textCol: 'task_text' };
    const docs = ragDocs({ cols, rows, target: 'completed' }, profile, []);

    expect(docs[0].text).toContain('task_text: Morning walk');
    expect(docs[0].text).toContain('due_date: 2026-11-23');
    expect(docs[0].text).not.toMatch(/task[_-]?id/i);
  });

  it('detects flow edges from source and destination columns', () => {
    const flowRows = trainingRows(12).map((row, i) => ({
      ...row,
      from_step: i % 2 ? 'review' : 'intake',
      to_step: i % 2 ? 'done' : 'review',
    }));
    const prepared = prepareDataset(Object.keys(flowRows[0]), flowRows, { target: 'completed' });
    const docs = ragDocs(prepared.dataset, prepared.profile, []);
    expect(docs.length).toBe(12);
    expect(docs.some((doc) => doc.edges.length > 0)).toBe(true);
  });
});
