import { describe, expect, it, vi } from 'vitest';
import type { RAGResult } from './engine';
import { buildAnswerContext, friendlyFallback, polishRagAnswer } from './answerEngine';
import type { ChatMessage } from './languageModel';

const rag: RAGResult = {
  mode: 'agentic',
  why: 'Multi-step retrieval',
  plan: ['Step 1: searched'],
  trail: '',
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
  it('lets the local assistant rewrite a static extract into a natural answer', async () => {
    const generator = vi.fn(async (_messages: ChatMessage[]) =>
      'Answer: Boating is spending time on the water in a boat. For this task, it is a personal item with medium priority.'
    );
    const stages: string[] = [];

    const result = await polishRagAnswer({
      question: 'What is this task about?',
      rag,
      facts: ['Task title: Boating.', 'Category: personal.', 'Priority: medium.'],
      generator,
      onProgress: (stage) => stages.push(stage),
    });

    expect(generator).toHaveBeenCalledOnce();
    expect(result.answer).toMatch(/on the water/i);
    expect(result.answer).not.toMatch(/^Answer:/i);
    expect(stages).toContain('Understanding your question…');
  });

  it('still explains the topic when the small model is unavailable', () => {
    const answer = friendlyFallback('What is this task about?', rag, [
      'Task title: Boating.',
      'Category: personal.',
      'Priority: medium.',
    ]);

    expect(answer).toMatch(/water/i);
    expect(answer).toMatch(/medium priority/i);
    expect(answer).not.toMatch(/task[_-]?id/i);
    expect(answer).not.toBe(rag.answer);
  });

  it('keeps exact completion probabilities in the fallback', () => {
    const answer = friendlyFallback('What is the completion probability of Boating?', rag, [
      'Completion probability: 99.9%.',
    ]);

    expect(answer).toContain('99.9%');
  });

  it('builds a compact context that keeps exact facts and drops ids', () => {
    const context = buildAnswerContext(rag, ['Due date: 2026-11-17.', 'task_id: T12']);

    expect(context).toContain('Due date: 2026-11-17.');
    expect(context).toContain('task_id: T12'); // retained only inside the retrieved excerpt
    expect(context.match(/task_id/g)).toHaveLength(1);
    expect(context.length).toBeLessThan(3201);

    const aboutContext = buildAnswerContext(rag, ['Task title: Boating.'], 'What is this task about?');
    expect(aboutContext).toContain('Topic hint:');
    expect(aboutContext).toMatch(/on the water/i);
  });

  it('falls back safely when generation throws', async () => {
    const result = await polishRagAnswer({
      question: 'What is this task about?',
      rag,
      facts: ['Task title: Boating.'],
      generator: async () => {
        throw new Error('model failed');
      },
    });

    expect(result.answer).toMatch(/water/i);
  });
});
