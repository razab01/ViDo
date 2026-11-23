/**
 * Client for the optional Python ML/Graph-RAG service.
 *
 * The React app still works without it: every exported helper throws when the
 * service is unavailable, and callers fall back to the browser-only engine.
 */
import type {
  Artifact,
  Dataset,
  Profile,
  Row,
  TrainingOptions,
  TrainingResult,
  PythonModelContext,
} from './engine';
import { pct } from './engine';

const BASE = '/api/python';
const predictionCache = new Map<string, number[]>();
let availability: { ok: boolean; at: number } | null = null;

const cacheKey = (context: PythonModelContext, model: string, row: Row): string =>
  JSON.stringify([
    context.run_id,
    model,
    context.cols.map((column) => String(row[column] ?? '')),
  ]);

/** Stable key for the React layer to know when an async inference has arrived. */
export const pythonPredictionKey = cacheKey;

async function post<T>(path: string, body: unknown, timeoutMs = 60000): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(String(detail.detail || `Python service returned ${response.status}`));
    }
    return (await response.json()) as T;
  } finally {
    window.clearTimeout(timer);
  }
}

export async function isPythonAvailable(): Promise<boolean> {
  // Positive checks stay cached for 30s. A failed check is retried quickly so
  // a slow-starting Python service does not force the browser fallback for
  // half a minute.
  const cacheMs = availability?.ok === false ? 2000 : 30000;
  if (availability && Date.now() - availability.at < cacheMs) return availability.ok;
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch(`${BASE}/health`, { signal: controller.signal });
    availability = { ok: response.ok, at: Date.now() };
    return response.ok;
  } catch {
    availability = { ok: false, at: Date.now() };
    return false;
  } finally {
    window.clearTimeout(timer);
  }
}

export function createPythonArtifact(context: PythonModelContext, model: string): Artifact {
  const uniform = Array.from({ length: context.labels.length }, () => 1 / Math.max(1, context.labels.length));
  return {
    name: model,
    proba: (row) => predictionCache.get(cacheKey(context, model, row)) ?? uniform,
    predict: (row) => {
      const values = predictionCache.get(cacheKey(context, model, row)) ?? uniform;
      let best = 0;
      for (let index = 1; index < values.length; index++) {
        if (values[index] > values[best]) best = index;
      }
      return best;
    },
  };
}

/** Fetch fresh probabilities for a model/row so the synchronous artifact stays correct. */
export async function warmPythonPrediction(
  result: TrainingResult,
  model: string,
  row: Row
): Promise<number[]> {
  const context = result.python;
  if (!context) return [];
  const response = await post<{ probabilities: number[] }>('/predict', { context, model, row }, 15000);
  predictionCache.set(cacheKey(context, model, row), response.probabilities);
  return response.probabilities;
}

interface PythonTrainResponse {
  labels: string[];
  features: string[];
  history: TrainingResult['history'];
  metrics: TrainingResult['metrics'];
  best: TrainingResult['best'];
  suggestions: TrainingResult['suggestions'];
  sample: Row;
  trainSize: number;
  testSize: number;
  splitNote: string;
  capped: boolean;
  tokenizerNote: string;
  tokenizerWarning: string;
  tokenizer: TrainingResult['tokenizer'];
  baseline: number;
  binned: boolean;
  context: PythonModelContext;
  sample_probs: number[];
}

export async function trainWithPython(
  dataset: Dataset,
  profile: Profile,
  options: TrainingOptions,
  onStatus: (message: string) => void,
  onProgress: (done: number, total: number, label: string) => void
): Promise<TrainingResult> {
  onStatus('Starting PyTorch training…');
  onProgress(0, 10, 'Preparing tensors');
  const response = await post<PythonTrainResponse>(
    '/train',
    {
      rows: dataset.rows,
      cols: dataset.cols,
      profile,
      target: options.target,
      textCol: options.textCol,
      epochs: options.epochs,
      // Fast deterministic embeddings keep the preview responsive; the Python
      // service still uses Transformers MiniLM when called with usePretrained.
      usePretrained: false,
    },
    180000
  );

  const context = response.context;
  predictionCache.set(cacheKey(context, context.best_model, response.sample), response.sample_probs);
  onProgress(10, 10, 'PyTorch models ready');
  onStatus('PyTorch models ready.');

  return {
    dataset,
    profile,
    target: options.target,
    textCol: options.textCol,
    labels: response.labels,
    features: response.features,
    history: response.history,
    metrics: response.metrics,
    models: response.metrics.map((metric) => ({
      metric,
      artifact: createPythonArtifact(context, metric.n),
    })),
    python: context,
    best: response.best,
    artifact: createPythonArtifact(context, context.best_model),
    suggestions: response.suggestions,
    sample: response.sample,
    trainSize: response.trainSize,
    testSize: response.testSize,
    tokenizerNote: response.tokenizerNote,
    tokenizerWarning: response.tokenizerWarning,
    tokenizer: response.tokenizer,
    baseline: response.baseline,
    binned: response.binned,
    splitNote: response.splitNote,
    capped: response.capped,
  };
}

export interface PythonAnswerInput {
  question: string;
  rows: Row[];
  cols: string[];
  profile: Profile;
  suggestions: TrainingResult['suggestions'];
  trained: {
    best: { n: string };
    features: string[];
    target: string;
    textCol: string;
  } | null;
}

export async function answerWithPython(input: PythonAnswerInput): Promise<string> {
  // The Python endpoint expects a dict for `trained`; normalize a missing
  // model to {} so a fresh chat session never gets a 422 and falls back.
  const response = await post<{ answer: string }>(
    '/answer',
    { ...input, trained: input.trained ?? {} },
    30000
  );
  return response.answer;
}

/** Display helper so the UI can say which backend produced the numbers. */
export const pythonBackendLabel = (context?: PythonModelContext): string =>
  context ? `PyTorch · ${context.device}` : 'browser engine';

export const pythonAccuracyLabel = (value: number): string => pct(value);
