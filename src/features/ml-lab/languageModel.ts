/**
 * Small on-device language model for ViDo.
 *
 * The model weights are downloaded from Hugging Face the first time the
 * assistant warms up (or answers a question), then cached by the browser. All
 * loading and inference happens in a Web Worker, so the page stays responsive
 * even while a few hundred megabytes are being downloaded and decoded.
 *
 * The library itself is installed with the app (`@huggingface/transformers`);
 * no API key or backend is needed.
 */

export interface ChatMessage {
  role: 'system' | 'user';
  content: string;
}

export type ModelProgress = (stage: string) => void;

/** Kept for callers that want to inject a generator directly. */
export interface LocalGenerator {
  (messages: ChatMessage[], onProgress?: ModelProgress): Promise<string>;
}

/** Worker commands without the request id (which the page assigns). */
export type ModelWorkerPayload =
  | { type: 'generate'; messages: ChatMessage[] }
  | { type: 'warm' };

/** Messages sent from the page to the model worker. */
export type ModelWorkerRequest = ModelWorkerPayload & { requestId: string };

/** Messages sent from the model worker back to the page. */
export type ModelWorkerResponse =
  | { type: 'stage'; requestId: string | '*'; stage: string }
  | { type: 'result'; requestId: string; text: string }
  | { type: 'error'; requestId: string; message: string };

const REQUEST_TIMEOUT_MS = 180_000;
const WARMUP_IDLE_TIMEOUT_MS = 2000;
const WARMUP_FALLBACK_DELAY_MS = 800;

let worker: Worker | null = null;
let requestCounter = 0;
let warmupScheduled = false;

interface PendingRequest {
  resolve: (text: string) => void;
  reject: (error: Error) => void;
  onProgress?: ModelProgress;
  timer: number | null;
}

const pending = new Map<string, PendingRequest>();

const canRun = (): boolean =>
  typeof window !== 'undefined' &&
  typeof document !== 'undefined' &&
  typeof Worker !== 'undefined' &&
  import.meta.env?.MODE !== 'test';

/** Restart the timeout whenever the worker reports progress. */
const armTimer = (requestId: string): void => {
  const request = pending.get(requestId);
  if (!request) return;
  if (request.timer !== null) window.clearTimeout(request.timer);
  request.timer = window.setTimeout(() => {
    pending.delete(requestId);
    request.reject(new Error('The local assistant is taking longer than expected. Try again.'));
  }, REQUEST_TIMEOUT_MS);
};

const failAll = (message: string): void => {
  pending.forEach((request) => {
    if (request.timer !== null) window.clearTimeout(request.timer);
    request.reject(new Error(message));
  });
  pending.clear();
};

const getWorker = (): Worker => {
  if (worker) return worker;
  try {
    const created = new Worker(new URL('./languageModel.worker.ts', import.meta.url), {
      type: 'module',
      name: 'vido-local-assistant',
    });

    created.onmessage = (event: MessageEvent<ModelWorkerResponse>) => {
      const message = event.data;
      if (!message || typeof message !== 'object') return;

      if (message.type === 'stage') {
        if (message.requestId === '*') {
          // Model download/load progress belongs to every waiting caller.
          pending.forEach((request, id) => {
            request.onProgress?.(message.stage);
            armTimer(id);
          });
        } else {
          const request = pending.get(message.requestId);
          if (request) {
            request.onProgress?.(message.stage);
            armTimer(message.requestId);
          }
        }
        return;
      }

      const request = pending.get(message.requestId);
      if (!request) return;
      if (request.timer !== null) window.clearTimeout(request.timer);
      pending.delete(message.requestId);

      if (message.type === 'result') request.resolve(message.text);
      else request.reject(new Error(message.message || 'The local assistant could not answer.'));
    };

    created.onerror = () => failAll('The local assistant stopped unexpectedly. Try again.');
    created.onmessageerror = () => failAll('The local assistant could not be reached. Try again.');

    worker = created;
    return created;
  } catch {
    throw new Error('The local assistant is not available in this browser.');
  }
};

const sendRequest = (
  payload: ModelWorkerPayload,
  onProgress?: ModelProgress
): Promise<string> => {
  const activeWorker = getWorker();
  const requestId = `req-${++requestCounter}`;
  const promise = new Promise<string>((resolve, reject) => {
    pending.set(requestId, { resolve, reject, onProgress, timer: null });
    armTimer(requestId);
  });
  activeWorker.postMessage({ ...payload, requestId } as ModelWorkerRequest);
  return promise;
};

/** Ask the local assistant to rewrite RAG notes into a natural answer. */
export async function generateLocalAnswer(
  messages: ChatMessage[],
  onProgress?: ModelProgress
): Promise<string> {
  if (!canRun()) throw new Error('The local assistant is not available in this environment.');
  return sendRequest({ type: 'generate', messages }, onProgress);
}

/**
 * Download and initialise the model without answering anything. Safe to call
 * repeatedly: only one warm-up is scheduled per page load.
 */
export async function warmLocalLanguageModel(): Promise<void> {
  if (!canRun()) return;
  await sendRequest({ type: 'warm' });
}

/**
 * Queue a background download when the browser is idle so the first real
 * question is already fast. Progress is forwarded to any active answer UI.
 */
export function scheduleLocalModelWarmup(): void {
  if (warmupScheduled || !canRun()) return;
  warmupScheduled = true;

  const run = () => {
    // Ask the browser to keep the cached model files even when storage is low.
    const storage = navigator.storage;
    if (storage?.persist) {
      void Promise.resolve(storage.persisted?.())
        .then((persisted) => {
          if (!persisted) void storage.persist?.();
        })
        .catch(() => undefined);
    }
    void warmLocalLanguageModel().catch(() => undefined);
  };

  const idle = (window as Window & {
    requestIdleCallback?: (callback: () => void, options?: { timeout: number }) => number;
  }).requestIdleCallback;

  if (typeof idle === 'function') idle.call(window, run, { timeout: WARMUP_IDLE_TIMEOUT_MS });
  else window.setTimeout(run, WARMUP_FALLBACK_DELAY_MS);
}

/** Useful for tests or if the app ever offers a model reset control. */
export function resetLocalLanguageModel(): void {
  failAll('The local assistant was reset.');
  worker?.terminate();
  worker = null;
  warmupScheduled = false;
}
