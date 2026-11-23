import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  generateLocalAnswer,
  resetLocalLanguageModel,
  scheduleLocalModelWarmup,
  warmLocalLanguageModel,
} from './languageModel';
import type { ModelWorkerRequest, ModelWorkerResponse } from './languageModel';

class MockWorker {
  static instances: MockWorker[] = [];

  onmessage: ((event: MessageEvent<ModelWorkerResponse>) => void) | null = null;
  onerror: (() => void) | null = null;
  onmessageerror: (() => void) | null = null;
  posted: ModelWorkerRequest[] = [];
  terminated = false;

  constructor() {
    MockWorker.instances.push(this);
  }

  postMessage(message: ModelWorkerRequest): void {
    this.posted.push(message);
  }

  terminate(): void {
    this.terminated = true;
  }

  emit(message: ModelWorkerResponse): void {
    this.onmessage?.({ data: message } as MessageEvent<ModelWorkerResponse>);
  }
}

const latestWorker = (): MockWorker => {
  const instance = MockWorker.instances[MockWorker.instances.length - 1];
  if (!instance) throw new Error('No worker was created.');
  return instance;
};

describe('local language model worker', () => {
  beforeEach(() => {
    vi.stubEnv('MODE', 'development');
    vi.stubGlobal('Worker', MockWorker);
    MockWorker.instances = [];
    resetLocalLanguageModel();
    vi.useFakeTimers();
  });

  afterEach(() => {
    resetLocalLanguageModel();
    vi.useRealTimers();
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it('sends generation to the worker and forwards its progress and answer', async () => {
    const stages: string[] = [];
    const answer = generateLocalAnswer(
      [{ role: 'user', content: 'Question: What is this task about?' }],
      (stage) => stages.push(stage)
    );

    const instance = latestWorker();
    const request = instance.posted[0];
    expect(request).toMatchObject({ type: 'generate' });
    expect(request.type).toBe('generate');
    if (request.type !== 'generate') throw new Error('Expected a generate request.');
    expect(request.messages[0].content).toContain('What is this task about?');

    instance.emit({ type: 'stage', requestId: '*', stage: 'Loading the small language model…' });
    instance.emit({ type: 'stage', requestId: request.requestId, stage: 'Writing a clear answer…' });
    instance.emit({ type: 'result', requestId: request.requestId, text: 'A natural answer.' });

    await expect(answer).resolves.toBe('A natural answer.');
    expect(stages).toEqual(['Loading the small language model…', 'Writing a clear answer…']);
  });

  it('rejects worker errors so the RAG layer can use its fallback', async () => {
    const answer = generateLocalAnswer([{ role: 'user', content: 'Question: hi' }]);
    const instance = latestWorker();
    const request = instance.posted[0];

    instance.emit({ type: 'error', requestId: request.requestId, message: 'model failed' });

    await expect(answer).rejects.toThrow('model failed');
  });

  it('reuses a single worker for repeated answers', async () => {
    const first = generateLocalAnswer([{ role: 'user', content: 'one' }]);
    const instance = latestWorker();
    const firstRequest = instance.posted[0];
    instance.emit({ type: 'result', requestId: firstRequest.requestId, text: 'one' });
    await first;

    const second = generateLocalAnswer([{ role: 'user', content: 'two' }]);
    expect(latestWorker()).toBe(instance);
    expect(instance.posted).toHaveLength(2);

    const secondRequest = instance.posted[1];
    instance.emit({ type: 'result', requestId: secondRequest.requestId, text: 'two' });
    await expect(second).resolves.toBe('two');
  });

  it('warms the model without generating an answer', async () => {
    const warming = warmLocalLanguageModel();
    const instance = latestWorker();
    const request = instance.posted[0];
    expect(request.type).toBe('warm');
    instance.emit({ type: 'result', requestId: request.requestId, text: '' });
    await expect(warming).resolves.toBeUndefined();
  });

  it('schedules only one background warm-up when the browser is idle', async () => {
    scheduleLocalModelWarmup();
    scheduleLocalModelWarmup();
    expect(MockWorker.instances).toHaveLength(0);

    await vi.advanceTimersByTimeAsync(2500);

    expect(MockWorker.instances).toHaveLength(1);
    expect(latestWorker().posted[0].type).toBe('warm');

    const request = latestWorker().posted[0];
    latestWorker().emit({ type: 'result', requestId: request.requestId, text: '' });

    scheduleLocalModelWarmup();
    await vi.advanceTimersByTimeAsync(2500);
    expect(MockWorker.instances).toHaveLength(1);
  });

  it('stays unavailable when workers are not supported', async () => {
    vi.stubGlobal('Worker', undefined);
    await expect(generateLocalAnswer([])).rejects.toThrow(/not available/i);
    expect(MockWorker.instances).toHaveLength(0);
  });
});
