import { describe, expect, it } from 'vitest';
import type { RAGResult } from './engine';
import { composeAnswer, friendlyFallback, polishRagAnswer } from './answerEngine';

const rag: RAGResult = {
  mode: 'graph',
  why: 'Graph retrieval connects your question to related records and built-in knowledge.',
  plan: [],
  trail: 'Graph seeds: boating · 18 nodes reached in 2 hops',
  ranked: [
    {
      title: 'Boating',
      score: 4.2,
      excerpt: 'task_id: T12 Task title: Boating. Category: personal. Priority: medium.',
    },
  ],
  answer: 'This task is “Boating” in the personal category.',
};

describe('answerEngine', () => {
  it('explains the subject from bundled knowledge instead of echoing raw fields', async () => {
    const result = await polishRagAnswer({
      question: 'What is this task about?',
      rag,
      facts: ['Task title: Boating.', 'Description: Visit parents.', 'Category: personal.'],
    });

    expect(result.mode).toBe('graph');
    expect(result.answer).toMatch(/Boating is spending time on the water/i);
    expect(result.answer).toMatch(/visit parents/i);
    expect(result.answer).not.toMatch(/task title\s*:/i);
  });

  it('explains a report title from all its words instead of a status-report keyword', () => {
    const answer = composeAnswer({
      question: 'What is this task about?',
      rag: { ...rag, ranked: [], answer: 'No relevant context found' },
      facts: [
        'Task title: financial report for this quarter.',
        'Priority: medium.',
        'Due date: 2026-11-23.',
      ],
    });

    expect(answer).toMatch(/clear summary of what happened/i);
    expect(answer).toMatch(/money, budgets and accounts/i);
    expect(answer).toMatch(/three-month period/i);
    expect(answer).not.toMatch(/status report/i);
  });

  it('uses a supplied Wikipedia summary when one is available', () => {
    const answer = composeAnswer({
      question: 'What is quantum computing?',
      rag: { ...rag, ranked: [], answer: 'No relevant context found' },
      facts: [],
      topic: {
        title: 'Quantum computing',
        extract: 'Quantum computing uses quantum mechanics to process information in new ways.',
      },
    });

    expect(answer).toContain('Quantum computing uses quantum mechanics to process information in new ways.');
  });

  it('keeps exact dates and times in natural sentences', () => {
    const answer = composeAnswer({
      question: 'When is this task due?',
      rag,
      facts: ['Task title: Boating.', 'Due date: 2026-11-23.', 'Due time: 23:59.'],
    });

    expect(answer).toContain('This task is due 2026-11-23 at 23:59.');
    expect(answer).not.toMatch(/^Due date\s*:/i);
  });

  it('preserves completion probabilities and explains what they mean', () => {
    const answer = composeAnswer({
      question: 'What is the completion probability of Boating?',
      rag,
      facts: ['Task title: Boating.', 'Completion probability: 99.9%.'],
    });

    expect(answer).toContain('99.9%');
    expect(answer).toMatch(/estimated completion probability/i);
    expect(answer).toMatch(/estimate|likely/i);
    expect(answer).not.toMatch(/^(Completion probability|Task title)\s*:/i);
  });

  it('gives a concrete next step using the task facts', () => {
    const answer = composeAnswer({
      question: 'What should I do next?',
      rag,
      facts: [
        'Task title: Boating.',
        'Priority: medium.',
        'Due date: 2026-11-25.',
        'Checkpoint: Task initialized in workspace.',
      ],
    });

    expect(answer).toMatch(/next step/i);
    expect(answer).toMatch(/medium priority/i);
    expect(answer).toContain('2026-11-25');
    expect(answer).toMatch(/task initialized in workspace/i);
  });

  it('answers general knowledge questions with no uploaded data', () => {
    const answer = composeAnswer({
      question: 'What is prioritisation?',
      rag: { ...rag, ranked: [], answer: 'No relevant context found' },
      facts: [],
    });

    expect(answer).toMatch(/deciding which work deserves attention first/i);
    expect(answer).not.toMatch(/no relevant context/i);
  });

  it('summarises the closest records for comparison questions', () => {
    const answer = composeAnswer({
      question: 'Which records look likely to be completed?',
      rag: {
        ...rag,
        ranked: [
          { title: 'Review budget', score: 3, excerpt: '' },
          { title: 'Plan a trip', score: 2, excerpt: '' },
        ],
      },
      facts: [],
    });

    expect(answer).toMatch(/closest records/i);
    expect(answer).toMatch(/review budget/i);
    expect(answer).toMatch(/plan a trip/i);
  });

  it('never returns a raw record dump, even when retrieval extracted one', () => {
    const answer = composeAnswer({
      question: 'What is this task about?',
      rag: { ...rag, answer: 'Task title: Boating task: Visit parents' },
      facts: ['Task title: Boating.', 'Description: Visit parents.'],
    });

    expect(answer).not.toMatch(/task title\s*:/i);
    expect(answer).not.toMatch(/task:\s*visit parents/i);
    expect(answer).toMatch(/on the water/i);
  });

  it('composes deterministically and reports short progress stages', async () => {
    const options = {
      question: 'What is this task about?',
      rag,
      facts: ['Task title: Boating.'],
    };
    const first = composeAnswer(options);
    const second = composeAnswer(options);
    expect(second).toBe(first);

    const stages: string[] = [];
    const result = await polishRagAnswer({ ...options, onProgress: (stage) => stages.push(stage) });
    expect(result.answer).toBe(first);
    expect(stages).toContain('Connecting related ideas…');
    expect(stages).toContain('Writing your answer…');
  });

  it('keeps the friendly fallback useful for unrelated questions', () => {
    const answer = friendlyFallback('How do I reorganise my week?', { ...rag, ranked: [] }, []);

    expect(answer.length).toBeGreaterThan(40);
    expect(answer).toMatch(/[.!?]$/);
    expect(answer).not.toMatch(/no relevant context/i);
  });
});
