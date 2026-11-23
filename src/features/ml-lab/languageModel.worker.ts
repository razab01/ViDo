/**
 * Web Worker for ViDo's on-device language model.
 *
 * Keeping model download, WASM initialisation and text generation here means
 * the React page never freezes, even on the very first answer.
 */

import type { ChatMessage, ModelWorkerRequest, ModelWorkerResponse } from './languageModel';

const MODEL_ID = 'onnx-community/SmolLM2-360M-Instruct-ONNX';

type Generator = (messages: ChatMessage[], options: Record<string, unknown>) => Promise<unknown>;

const scope = self as unknown as {
  postMessage: (message: ModelWorkerResponse) => void;
  onmessage: ((event: MessageEvent<ModelWorkerRequest>) => void) | null;
};

let generatorPromise: Promise<Generator> | null = null;
let queue: Promise<void> = Promise.resolve();

const post = (message: ModelWorkerResponse): void => scope.postMessage(message);

const friendlyDownloadStage = (payload: unknown): string | null => {
  const info = payload as { status?: string; progress?: number } | null;
  if (!info) return null;
  const status = String(info.status || '');
  if (status === 'done' || status === 'ready') return 'Getting the assistant ready…';
  if (status === 'progress' || status === 'download') {
    const pct = typeof info.progress === 'number' ? ` · ${Math.round(info.progress)}%` : '';
    return `Getting the assistant ready${pct}`;
  }
  return null;
};

const extractGeneratedText = (raw: unknown): string => {
  const first = Array.isArray(raw) ? raw[0] : raw;
  if (!first) return '';
  const record = first as { generated_text?: unknown };
  const generated = record.generated_text ?? first;

  if (Array.isArray(generated)) {
    const last = generated[generated.length - 1] as { content?: unknown } | undefined;
    if (last && typeof last.content === 'string') return last.content;
  }
  if (typeof generated === 'string') return generated;
  if (generated && typeof generated === 'object') {
    const obj = generated as { content?: unknown; text?: unknown };
    if (typeof obj.content === 'string') return obj.content;
    if (typeof obj.text === 'string') return obj.text;
  }
  return '';
};

const loadGenerator = async (): Promise<Generator> => {
  if (generatorPromise) return generatorPromise;

  generatorPromise = (async () => {
    post({ type: 'stage', requestId: '*', stage: 'Waking up the assistant…' });
    const { pipeline, env } = await import('@huggingface/transformers');

    // A single WASM thread works without special cross-origin isolation headers.
    const runtime = env as unknown as {
      allowLocalModels?: boolean;
      backends?: { onnx?: { wasm?: { numThreads?: number } } };
    };
    runtime.allowLocalModels = false;
    if (runtime.backends?.onnx?.wasm) runtime.backends.onnx.wasm.numThreads = 1;

    post({ type: 'stage', requestId: '*', stage: 'Loading the small language model…' });
    const generator = await pipeline('text-generation', MODEL_ID, {
      dtype: 'q4',
      device: 'wasm',
      progress_callback: (payload: unknown) => {
        const stage = friendlyDownloadStage(payload);
        if (stage) post({ type: 'stage', requestId: '*', stage });
      },
    } as never);

    return generator as unknown as Generator;
  })();

  try {
    return await generatorPromise;
  } catch (error) {
    generatorPromise = null;
    throw error;
  }
};

const handleRequest = async (request: ModelWorkerRequest): Promise<void> => {
  try {
    const generator = await loadGenerator();

    if (request.type === 'warm') {
      post({ type: 'result', requestId: request.requestId, text: '' });
      return;
    }

    post({ type: 'stage', requestId: request.requestId, stage: 'Writing a clear answer…' });
    const output = await generator(request.messages, {
      max_new_tokens: 140,
      do_sample: false,
      repetition_penalty: 1.05,
    });
    post({ type: 'result', requestId: request.requestId, text: extractGeneratedText(output) });
  } catch (error) {
    post({
      type: 'error',
      requestId: request.requestId,
      message: error instanceof Error ? error.message : String(error),
    });
  }
};

scope.onmessage = (event: MessageEvent<ModelWorkerRequest>) => {
  const request = event.data;
  // Serialise jobs: one cached generator, one inference at a time.
  queue = queue.then(() => handleRequest(request));
};
