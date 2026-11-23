/**
 * Per-user ML Lab persistence.
 *
 * The trained artifact contains closures, so it is kept in memory and rebuilt
 * from the saved dataset/options when a page is reloaded. Everything else is
 * serialized to localStorage under a key scoped to the signed-in user.
 */
import { formatConfidence, syncPredictionRow, trainAll } from './engine';
import type {
  Dataset,
  PreparedData,
  Profile,
  Row,
  TrainingOptions,
  TrainingResult,
  UploadedData,
} from './engine';

export type SerializableTrainingResult = Omit<TrainingResult, 'artifact'> & {
  artifactName: string;
};

export interface StoredUpload {
  name: string;
  names: string[];
  sheets: Record<string, { cols: string[]; rows: Row[] }>;
  sheet: string;
  data: PreparedData;
}

export interface StoredMLLabState {
  version: 1;
  userId: string;
  savedAt: number;
  upload: StoredUpload | null;
  sheetName: string;
  dataset: Dataset | null;
  profile: Profile;
  target: string;
  textCol: string;
  dateCol: string;
  tokenizer: TrainingOptions['tokenizer'];
  split: TrainingOptions['split'];
  vocab: number;
  epochs: number;
  options: TrainingOptions | null;
  result: SerializableTrainingResult | null;
  predictionValues: Row;
}

export interface TaskPredictionInput {
  title: string;
  description?: string;
  priority?: string;
  category?: string;
  startDate?: string;
  dueDate?: string;
  dueTime?: string;
}

export interface TaskPrediction {
  label: string;
  probability: number;
  model: string;
  target: string;
  kind: 'confidence' | 'test accuracy';
  isCompletion: boolean;
}

const STORAGE_PREFIX = 'vido_ml_lab_v1_';

const snapshots = new Map<string, StoredMLLabState | null>();
const activeResults = new Map<string, TrainingResult>();
const snapshotListeners = new Map<string, Set<() => void>>();
const resultListeners = new Map<string, Set<(result: TrainingResult | null) => void>>();
const restoreGenerations = new Map<string, number>();
const restores = new Map<string, Promise<void>>();

const keyFor = (userId: string) => STORAGE_PREFIX + encodeURIComponent(userId || 'anonymous');

function notifySnapshot(userId: string) {
  snapshotListeners.get(userId)?.forEach((listener) => listener());
}

function notifyResult(userId: string, result: TrainingResult | null) {
  resultListeners.get(userId)?.forEach((listener) => listener(result));
}

export function readSnapshot(userId: string): StoredMLLabState | null {
  if (snapshots.has(userId)) return snapshots.get(userId) ?? null;
  try {
    const raw = localStorage.getItem(keyFor(userId));
    if (raw) {
      const parsed = JSON.parse(raw) as StoredMLLabState;
      if (parsed.version === 1 && parsed.userId === userId) {
        snapshots.set(userId, parsed);
        return parsed;
      }
    }
  } catch {
    // Storage can be disabled or contain older/corrupt data.
  }
  snapshots.set(userId, null);
  return null;
}

export function saveSnapshot(
  userId: string,
  state: Omit<StoredMLLabState, 'version' | 'userId' | 'savedAt'>
) {
  const stored: StoredMLLabState = {
    ...state,
    version: 1,
    userId,
    savedAt: Date.now(),
  };
  snapshots.set(userId, stored);
  try {
    localStorage.setItem(keyFor(userId), JSON.stringify(stored));
  } catch {
    // Large workbooks may exceed the quota. Keep the dataset/model and drop
    // the raw workbook copy so the rest of the state still survives reloads.
    try {
      localStorage.setItem(keyFor(userId), JSON.stringify({ ...stored, upload: null }));
    } catch {
      // Still keep the in-memory copy for this tab.
    }
  }
  notifySnapshot(userId);
}

export function clearSnapshot(userId: string) {
  snapshots.set(userId, null);
  try {
    localStorage.removeItem(keyFor(userId));
  } catch {
    // Ignore storage errors.
  }
  cancelRestore(userId);
  clearActiveResult(userId);
  notifySnapshot(userId);
}

export function subscribeSnapshot(userId: string, listener: () => void): () => void {
  const listeners = snapshotListeners.get(userId) ?? new Set();
  listeners.add(listener);
  snapshotListeners.set(userId, listeners);
  return () => listeners.delete(listener);
}

export function toSerializableResult(result: TrainingResult): SerializableTrainingResult {
  const { artifact, ...rest } = result;
  return { ...rest, artifactName: artifact.name };
}

export function resultFromStored(
  stored: SerializableTrainingResult | null | undefined,
  artifact?: TrainingResult | null
): TrainingResult | null {
  if (!stored) return null;
  const { artifactName, ...rest } = stored;
  const usableArtifact = artifact?.artifact ?? {
    name: artifactName,
    predict: () => 0,
  };
  return { ...rest, artifact: usableArtifact } as TrainingResult;
}

export function getActiveResult(userId: string): TrainingResult | null {
  return activeResults.get(userId) ?? null;
}

export function setActiveResult(userId: string, result: TrainingResult | null) {
  if (result) activeResults.set(userId, result);
  else activeResults.delete(userId);
  notifyResult(userId, result);
}

export function clearActiveResult(userId: string) {
  setActiveResult(userId, null);
}

export function subscribeActiveResult(
  userId: string,
  listener: (result: TrainingResult | null) => void
): () => void {
  const listeners = resultListeners.get(userId) ?? new Set();
  listeners.add(listener);
  resultListeners.set(userId, listeners);
  return () => listeners.delete(listener);
}

export function cancelRestore(userId: string) {
  restoreGenerations.set(userId, (restoreGenerations.get(userId) ?? 0) + 1);
  restores.delete(userId);
}

/**
 * Rebuild the in-memory best model from a saved dataset/options pair. The
 * result summary is already persisted, so this runs quietly and only upgrades
 * probability predictions once it finishes.
 */
export function requestRestore(userId: string): Promise<void> {
  const snapshot = readSnapshot(userId);
  if (!snapshot?.result || !snapshot.options || !snapshot.dataset) return Promise.resolve();
  if (activeResults.has(userId) || restores.has(userId)) {
    return restores.get(userId) ?? Promise.resolve();
  }
  const generation = (restoreGenerations.get(userId) ?? 0) + 1;
  restoreGenerations.set(userId, generation);
  const promise = (async () => {
    try {
      const rebuilt = await trainAll(
        snapshot.dataset as Dataset,
        snapshot.profile,
        snapshot.options as TrainingOptions,
        () => undefined,
        () => undefined
      );
      if (restoreGenerations.get(userId) === generation) setActiveResult(userId, rebuilt);
    } catch {
      // The saved summary remains available even if rebuilding is impossible.
    } finally {
      if (restoreGenerations.get(userId) === generation) restores.delete(userId);
    }
  })();
  restores.set(userId, promise);
  return promise;
}

export function toStoredUpload(upload: UploadedData | null): StoredUpload | null {
  if (!upload) return null;
  return {
    name: upload.name,
    names: upload.names,
    sheets: upload.sheets,
    sheet: upload.sheet,
    data: upload.data,
  };
}

const findColumn = (cols: string[], pattern: RegExp) => cols.find((column) => pattern.test(column));

export const isCompletionTarget = (target: string): boolean =>
  /complet|done|finish|state|status|success|approved|won|churn/i.test(target || '');

/**
 * Format a model probability for display. A single decimal keeps distinct
 * estimates visible (e.g. 98.8% vs 99.6%) instead of rounding both to 100%.
 * Confidence values are capped just below 100% because a fitted classifier
 * should not be presented as absolutely certain.
 */
export function formatTaskProbability(value: number, kind: TaskPrediction['kind'] = 'confidence'): string {
  if (kind === 'confidence') return formatConfidence(value);
  if (!Number.isFinite(value)) return '—';
  return `${Math.round(Math.min(Math.max(value, 0), 1) * 100)}%`;
}

/** Build a probability estimate for a task from the user's winning model. */
export function predictTask(userId: string, input: TaskPredictionInput): TaskPrediction | null {
  const result = getActiveResult(userId);
  if (!result) return null;
  const cols = result.dataset.cols;
  const row: Row = { ...(result.sample ?? {}) };
  const text = [input.title, input.description].filter(Boolean).join(' ').trim();
  row[result.textCol] = text;

  const priority = findColumn(cols, /^priority$/i);
  if (priority && input.priority) row[priority] = input.priority;
  const category = findColumn(cols, /^(category|scope|type)$/i);
  if (category && input.category) row[category] = input.category;
  const startDate = findColumn(cols, /^(start|created|added).*(date|time)$|^start_date$/i);
  if (startDate && input.startDate) row[startDate] = input.startDate;
  const dueDate = findColumn(cols, /(due|deadline|end).*(date|time)|^date$|^due_date$/i);
  if (dueDate && input.dueDate) row[dueDate] = input.dueDate;

  let predicted = 0;
  let proba: number | null = null;
  try {
    const synced = syncPredictionRow(result, row);
    predicted = result.artifact.predict(synced);
    const values = result.artifact.proba?.(synced);
    if (values && Number.isFinite(values[predicted])) proba = values[predicted];
  } catch {
    return null;
  }

  const probability = proba ?? result.best.acc;
  return {
    label: result.labels[predicted] ?? String(predicted),
    probability: Number.isFinite(probability) ? probability : 0,
    model: result.artifact.name,
    target: result.target,
    kind: proba === null ? 'test accuracy' : 'confidence',
    isCompletion: isCompletionTarget(result.target),
  };
}
