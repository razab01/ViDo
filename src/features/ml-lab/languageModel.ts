/**
 * Small on-device language model for ViDo.
 *
 * The model weights are downloaded from Hugging Face the first time an answer
 * is requested, then cached by the browser. The library itself is installed
 * with the app (`@huggingface/transformers`); no API key or backend is needed.
 */

export interface ChatMessage {
  role: 'system' | 'user';
  content: string;
}

export type ModelProgress = (stage: string) => void;

export interface LocalGenerator {
  (messages: ChatMessage[], onProgress?: ModelProgress): Promise<string>;
}

const MODEL_ID = 'onnx-community/SmolLM2-360M-Instruct-ONNX';

let generatorPromise: Promise<LocalGenerator> | null = null;

const canRun = (): boolean =>
  typeof window !== 'undefined' &&
  typeof document !== 'undefined' &&
  import.meta.env?.MODE !== 'test';

const friendlyDownloadStage = (payload: unknown): string | null => {
  const info = payload as { status?: string; progress?: number; file?: string } | null;
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

async function loadGenerator(onProgress?: ModelProgress): Promise<LocalGenerator> {
  if (!canRun()) throw new Error('The local assistant is not available in this environment.');
  if (generatorPromise) return generatorPromise;

  generatorPromise = (async () => {
    onProgress?.('Waking up the assistant…');
    const { pipeline, env } = await import('@huggingface/transformers');

    // Single WASM thread works without special cross-origin isolation headers.
    const runtime = env as unknown as {
      allowLocalModels?: boolean;
      backends?: { onnx?: { wasm?: { numThreads?: number } } };
    };
    runtime.allowLocalModels = false;
    if (runtime.backends?.onnx?.wasm) runtime.backends.onnx.wasm.numThreads = 1;

    onProgress?.('Loading the small language model…');
    const generator = await pipeline('text-generation', MODEL_ID, {
      dtype: 'q4',
      device: 'wasm',
      progress_callback: (payload: unknown) => {
        const stage = friendlyDownloadStage(payload);
        if (stage) onProgress?.(stage);
      },
    } as never);

    return async (messages, generateProgress) => {
      generateProgress?.('Writing a clear answer…');
      const output = await generator(messages as never, {
        max_new_tokens: 140,
        do_sample: false,
        repetition_penalty: 1.05,
      } as never);
      return extractGeneratedText(output);
    };
  })();

  try {
    return await generatorPromise;
  } catch (error) {
    generatorPromise = null;
    throw error;
  }
}

export async function generateLocalAnswer(
  messages: ChatMessage[],
  onProgress?: ModelProgress
): Promise<string> {
  const generator = await loadGenerator(onProgress);
  return generator(messages, onProgress);
}

/** Useful for tests or if the app ever offers a model reset control. */
export function resetLocalLanguageModel(): void {
  generatorPromise = null;
}
