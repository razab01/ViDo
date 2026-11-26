import React, { useEffect, useMemo, useState } from 'react';
import { AlertCircle, BarChart3, CheckCircle2, Database, Sparkles, Upload } from 'lucide-react';
import { Badge, Button, Card, Input } from '../../components/ui';
import {
  dateNum,
  demoRows,
  findDeadlineColumn,
  formatConfidence,
  loadUploadedFile,
  pct,
  predictWith,
  prepareDataset,
  profileData,
  selectUploadSheet,
  sparkline,
  syncPredictionRow,
  trainAll,
} from './engine';
import type {
  Dataset,
  PreparedData,
  Profile,
  Row,
  TrainingOptions,
  TrainingResult,
  UploadedData,
} from './engine';
import {
  cancelRestore,
  clearActiveResult,
  getActiveResult,
  hasDatasetUploaded,
  markDatasetUploaded,
  readSnapshot,
  requestRestore,
  resultFromStored,
  saveSnapshot,
  setActiveResult,
  subscribeActiveResult,
  toSerializableResult,
  toStoredUpload,
} from './mlStore';
import type { StoredMLLabState } from './mlStore';
import {
  isPythonAvailable,
  pythonBackendLabel,
  pythonPredictionKey,
  trainWithPython,
  warmPythonPrediction,
} from './pythonApi';
import { TrainingProgress } from './WorkingIndicator';

export interface MLLabPageProps {
  userId?: string;
}

type TokenizerChoice = TrainingOptions['tokenizer'];
type SplitChoice = TrainingOptions['split'];

const errorMessage = (error: unknown): string =>
  error && typeof error === 'object' && 'message' in error
    ? String((error as Error).message)
    : String(error || 'Unknown error');

const pretty = (value: string): string => value.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

export const MLLabPage: React.FC<MLLabPageProps> = ({ userId = 'demo-user-uid' }) => {
  const [initialSnapshot] = useState(() => readSnapshot(userId));
  const [status, setStatus] = useState(() =>
    initialSnapshot
      ? `Saved dataset restored for this account.${initialSnapshot.result ? ' Preparing the saved model…' : ''}`
      : 'Upload a file to get started.'
  );
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState({ done: 0, total: 10, label: '' });
  const [savedAt, setSavedAt] = useState(initialSnapshot?.savedAt ?? 0);

  const [upload, setUpload] = useState<UploadedData | null>(
    () => (initialSnapshot?.upload as UploadedData | null) ?? null
  );
  const [datasetLoaded, setDatasetLoaded] = useState(() => hasDatasetUploaded(userId));
  const [sheetName, setSheetName] = useState(initialSnapshot?.sheetName ?? '');
  const [dataset, setDataset] = useState<Dataset | null>(initialSnapshot?.dataset ?? null);
  const [profile, setProfile] = useState<Profile>(() => initialSnapshot?.profile ?? profileData([], []));

  const [target, setTarget] = useState(initialSnapshot?.target ?? '');
  const [textCol, setTextCol] = useState(initialSnapshot?.textCol ?? '');
  const [dateCol, setDateCol] = useState(initialSnapshot?.dateCol ?? '');
  const [tokenizer, setTokenizer] = useState<TokenizerChoice>(initialSnapshot?.tokenizer ?? 'cl100k');
  const [split, setSplit] = useState<SplitChoice>(initialSnapshot?.split ?? 'auto');
  const [vocab, setVocab] = useState(initialSnapshot?.vocab ?? 2048);
  // Every model now trains for exactly five epochs; this is not adjustable.
  const epochs = 5;

  const [result, setResult] = useState<TrainingResult | null>(() =>
    resultFromStored(initialSnapshot?.result ?? null, getActiveResult(userId))
  );
  const [modelReady, setModelReady] = useState(() => Boolean(getActiveResult(userId)));
  const [predictionValues, setPredictionValues] = useState<Row>(initialSnapshot?.predictionValues ?? {});
  const [prediction, setPrediction] = useState<{ label: string; confidence: string; model: string } | null>(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [deadlineTask, setDeadlineTask] = useState('');
  const [deadlineDate, setDeadlineDate] = useState('');
  const [backendLabel, setBackendLabel] = useState('browser engine');
  const [warmedKey, setWarmedKey] = useState('');

  const [showAdvanced, setShowAdvanced] = useState(false);

  const hydrate = (snapshot: StoredMLLabState | null, active: TrainingResult | null) => {
    setUpload((snapshot?.upload as UploadedData | null) ?? null);
    setSheetName(snapshot?.sheetName ?? '');
    setDataset(snapshot?.dataset ?? null);
    setProfile(snapshot?.profile ?? profileData([], []));
    setTarget(snapshot?.target ?? '');
    setTextCol(snapshot?.textCol ?? '');
    setDateCol(snapshot?.dateCol ?? '');
    setTokenizer(snapshot?.tokenizer ?? 'cl100k');
    setSplit(snapshot?.split ?? 'auto');
    setVocab(snapshot?.vocab ?? 2048);
    const restored = resultFromStored(snapshot?.result ?? null, active);
    setResult(restored);
    setBackendLabel(pythonBackendLabel(restored?.python));
    setModelReady(Boolean(active));
    setPredictionValues(snapshot?.predictionValues ?? {});
    setPrediction(null);
    setSelectedModel('');
    setDeadlineTask('');
    setDeadlineDate('');
    setSavedAt(snapshot?.savedAt ?? 0);
    setProgress({ done: 0, total: 10, label: '' });
  };

  useEffect(() => {
    const snapshot = readSnapshot(userId);
    const active = getActiveResult(userId);
    cancelRestore(userId);
    hydrate(snapshot, active);
    if (snapshot) {
      setStatus(
        snapshot.result
          ? 'Saved dataset and model found for this account. Preparing predictions…'
          : 'Saved dataset restored for this account.'
      );
    }
    if (snapshot?.result && !active) void requestRestore(userId);
  }, [userId]);

  useEffect(
    () =>
      subscribeActiveResult(userId, (active) => {
        if (!active) return;
        setResult(active);
        setBackendLabel(pythonBackendLabel(active.python));
        setModelReady(true);
        setStatus('Saved model ready.');
      }),
    [userId]
  );

  useEffect(() => {
    if (!result) {
      setSelectedModel('');
      setDeadlineTask('');
      setDeadlineDate('');
      return;
    }
    setSelectedModel((current) =>
      result.metrics.some((metric) => metric.n === current) ? current : result.best.n
    );
    const sample = result.sample ?? {};
    setDeadlineTask(String(sample[result.textCol] ?? ''));
    const deadlineCol = findDeadlineColumn(result.dataset.cols);
    setDeadlineDate(deadlineCol ? String(sample[deadlineCol] ?? '').slice(0, 10) : '');
  }, [result]);

  useEffect(() => {
    if (!dataset && !upload) return;
    const timer = window.setTimeout(() => {
      const options: TrainingOptions | null = result
        ? {
            target: result.target,
            textCol: result.textCol,
            tokenizer,
            split,
            vocab,
            epochs,
            bpeEmb: 16,
            bpeHidden: 24,
            bpeEpochs: epochs,
          }
        : null;
      saveSnapshot(userId, {
        upload: toStoredUpload(upload),
        sheetName,
        dataset,
        profile,
        target,
        textCol,
        dateCol,
        tokenizer,
        split,
        vocab,
        epochs,
        options,
        result: result ? toSerializableResult(result) : null,
        predictionValues,
      });
      setSavedAt(Date.now());
    }, 350);
    return () => window.clearTimeout(timer);
  }, [
    userId,
    upload,
    sheetName,
    dataset,
    profile,
    target,
    textCol,
    dateCol,
    tokenizer,
    split,
    vocab,
    epochs,
    result,
    predictionValues,
  ]);

  const columns = dataset?.cols ?? [];
  const dateOptions = columns.filter((column) => profile.P[column]?.isDate || profile.dates?.includes(column));

  const runTraining = async (source: Dataset, sourceProfile: Profile, startMessage: string) => {
    cancelRestore(userId);
    setTraining(true);
    setError('');
    setPrediction(null);
    setProgress({ done: 0, total: 10, label: 'Starting…' });
    setStatus(startMessage);
    try {
      const activeProfile: Profile = {
        ...sourceProfile,
        target: source.target || sourceProfile.target,
        textCol: sourceProfile.textCol || source.cols[0] || '',
        primaryDate: sourceProfile.primaryDate || null,
      };
      const activeDataset: Dataset = { ...source, target: activeProfile.target };
      const options: TrainingOptions = {
        target: activeProfile.target,
        textCol: activeProfile.textCol || '',
        tokenizer,
        split,
        vocab,
        epochs,
        bpeEmb: 16,
        bpeHidden: 24,
        bpeEpochs: epochs,
      };
      let trained: TrainingResult;
      if (await isPythonAvailable()) {
        try {
          trained = await trainWithPython(
            activeDataset,
            activeProfile,
            options,
            (message) => setStatus(message),
            (done, total, label) => setProgress({ done, total, label })
          );
          setBackendLabel(pythonBackendLabel(trained.python));
        } catch (pythonError) {
          setStatus('Python backend unavailable here — using the browser engine.');
          trained = await trainAll(
            activeDataset,
            activeProfile,
            options,
            () => undefined,
            (done, total, label) => setProgress({ done, total, label })
          );
          setBackendLabel('browser engine');
          console.warn('[ViDo] Python training failed; used browser fallback.', pythonError);
        }
      } else {
        trained = await trainAll(
          activeDataset,
          activeProfile,
          options,
          () => undefined,
          (done, total, label) => setProgress({ done, total, label })
        );
        setBackendLabel('browser engine');
      }
      setResult(trained);
      setActiveResult(userId, trained);
      setModelReady(true);
      setProfile(trained.profile);
      setTarget(trained.target);
      setTextCol(trained.textCol);
      setStatus('Ready. Predictions are available.');
      const initial: Row = {};
      Array.from(new Set([trained.textCol, ...trained.features])).forEach((column) => {
        initial[column] = String(trained.sample[column] ?? '');
      });
      setPredictionValues(initial);
    } catch (cause) {
      setError(errorMessage(cause));
      setStatus('Could not learn from that file. Try another file.');
    } finally {
      setTraining(false);
    }
  };

  const applyData = (data: PreparedData, message: string) => {
    if (!data.valid) {
      setError(data.message);
      setStatus(data.message);
      return;
    }
    clearActiveResult(userId);
    markDatasetUploaded(userId);
    setDatasetLoaded(true);
    setDataset(data.dataset);
    setProfile(data.profile);
    setTarget(data.profile.target || data.dataset.target || '');
    setTextCol(data.profile.textCol || data.dataset.cols[0] || '');
    setDateCol(data.profile.primaryDate || '');
    setResult(null);
    setModelReady(false);
    setPredictionValues({});
    setPrediction(null);
    setProgress({ done: 0, total: 10, label: '' });
    setStatus(message);
    setError('');
    void runTraining(data.dataset, data.profile, message);
  };

  const handleFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setError('');
    try {
      const uploaded = await loadUploadedFile(file, () => undefined);
      setUpload(uploaded);
      setSheetName(uploaded.sheet);
      applyData(uploaded.data, 'File received. Learning from it now…');
    } catch (cause) {
      setError(errorMessage(cause));
      setStatus('Could not read that file.');
    } finally {
      setBusy(false);
      event.target.value = '';
    }
  };

  const handleSheetChange = (name: string) => {
    if (!upload) return;
    try {
      const next = selectUploadSheet(upload, name);
      setUpload(next);
      setSheetName(next.sheet);
      applyData(next.data, 'Worksheet changed. Learning again…');
    } catch (cause) {
      setError(errorMessage(cause));
    }
  };

  const loadDemo = () => {
    const rows = demoRows();
    const table = prepareDataset(Object.keys(rows[0]), rows);
    setUpload(null);
    setSheetName('Demo dataset');
    applyData(table, 'Sample data loaded. Learning from it now…');
  };

  const handleTrain = () => {
    if (!dataset) {
      setError('Upload a file first.');
      return;
    }
    void runTraining(dataset, profile, 'Learning again…');
  };

  const predictionFields = useMemo(
    () => (result ? Array.from(new Set([result.textCol, ...result.features])) : []),
    [result]
  );

  const handlePredict = async () => {
    if (!result) return;
    const row = syncPredictionRow(result, predictionValues);
    if (result.python) {
      try {
        await warmPythonPrediction(result, result.artifact.name, row);
      } catch (pythonError) {
        setError(`Python inference unavailable — showing the last cached result. ${errorMessage(pythonError)}`);
      }
    }
    setPrediction(predictWith(result, row));
  };

  const progressPercent = Math.min(100, Math.round((progress.done / Math.max(1, progress.total)) * 100));
  const savedLabel = savedAt
    ? new Date(savedAt).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
    : '';
  // Model scores are only meaningful for data loaded in this session. A
  // persisted result is restored for instant predictions, but its percentages
  // stay hidden until the user uploads a file or loads the demo dataset.
  const metricText = (value: number) => (datasetLoaded ? pct(value) : '—');
  const lossMax = result ? Math.max(0.1, ...result.history.lt, ...result.history.le) : 1;
  const accuracyMax = result ? Math.max(0.1, ...result.history.at, ...result.history.ae) : 1;

  const deadlineColumn = result ? findDeadlineColumn(result.dataset.cols) : undefined;
  const selectableMetrics = result ? result.metrics.filter((metric) => !metric.err) : [];
  const completionLabelIndex = result
    ? result.labels.findIndex((label) => /^(completed|yes)$/i.test(label))
    : -1;
  const selectedArtifact =
    result?.models.find((model) => model.metric.n === selectedModel)?.artifact ?? result?.artifact;
  const pythonContext = result?.python;

  // Build once so the async Python warm-up and the displayed probability use
  // exactly the same derived row.
  const deadlineRow = useMemo<Row | null>(() => {
    if (!datasetLoaded || !result) return null;
    const row: Row = { ...(result.sample ?? {}) };
    row[result.textCol] = deadlineTask;
    if (deadlineColumn && deadlineDate) row[deadlineColumn] = deadlineDate;
    try {
      const synced = syncPredictionRow(result, row);
      if (deadlineColumn && deadlineDate) {
        const derived = 'days_to_' + (deadlineColumn.replace(/_date$/, '') || 'event');
        const startCol = (result.profile.dates || []).find(
          (dateCol) => dateCol !== deadlineColumn && dateCol !== result.profile.targetSource
        );
        if (result.dataset.cols.includes(derived) && startCol) {
          const start = dateNum(row[startCol]);
          const end = dateNum(row[deadlineColumn]);
          if (!isNaN(start) && !isNaN(end)) synced[derived] = String(Math.round(end - start));
        }
      }
      return synced;
    } catch {
      return row;
    }
  }, [datasetLoaded, result, deadlineTask, deadlineColumn, deadlineDate]);

  // Python owns inference: refresh the cache for the selected model/row before
  // the synchronous artifact is read by the probability card.
  const deadlineKey =
    pythonContext && deadlineRow && selectedModel
      ? pythonPredictionKey(pythonContext, selectedModel, deadlineRow)
      : '';
  useEffect(() => {
    if (!result || !pythonContext || !deadlineRow || !selectedModel || !deadlineKey) return;
    let cancelled = false;
    void warmPythonPrediction(result, selectedModel, deadlineRow)
      .then(() => {
        if (!cancelled) setWarmedKey(deadlineKey);
      })
      .catch(() => {
        // Keep the probability blank when Python cannot answer; never show a
        // uniform placeholder as if it were a trained estimate.
        if (!cancelled) setWarmedKey('');
      });
    return () => {
      cancelled = true;
    };
  }, [result, pythonContext, deadlineRow, selectedModel, deadlineKey]);

  const deadlineProbability = useMemo(() => {
    if (!deadlineRow || !result || !selectedArtifact?.proba || completionLabelIndex < 0) return null;
    if (pythonContext && warmedKey !== deadlineKey) return null;
    const values = selectedArtifact.proba(deadlineRow);
    return Number.isFinite(values[completionLabelIndex]) ? values[completionLabelIndex] : null;
  }, [deadlineRow, result, selectedArtifact, completionLabelIndex, pythonContext, warmedKey, deadlineKey]);

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-[1500px] mx-auto space-y-5">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Badge variant="accent" size="sm">Simple mode</Badge>
            <Badge variant={backendLabel.startsWith('PyTorch') ? 'success' : 'neutral'} size="sm">
              {backendLabel}
            </Badge>
            {savedLabel && (
              <Badge variant="success" size="sm">Saved · {savedLabel}</Badge>
            )}
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-[var(--ink)]">Learn from your file</h1>
          <p className="text-sm text-[var(--muted)] max-w-2xl">
            Upload a CSV or spreadsheet. ViDo works out the columns, learns from the data, and lets you ask
            questions about it — no setup needed.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowAdvanced((value) => !value)}
            aria-expanded={showAdvanced}
          >
            {showAdvanced ? 'Hide advanced options' : 'Advanced options'}
          </Button>
          <Button variant="secondary" size="sm" onClick={loadDemo} disabled={busy || training}>
            <Sparkles className="w-4 h-4" /> Load demo data
          </Button>
          <label className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-sm bg-[var(--accent)] text-white font-medium shadow-sm hover:opacity-90 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed">
            <Upload className="w-4 h-4" />
            {busy ? 'Reading…' : 'Upload file'}
            <input
              type="file"
              accept=".csv,.tsv,.txt,.dat,.xlsx,.xlsm,.xls,.ods"
              onChange={handleFile}
              disabled={busy || training}
              className="hidden"
              aria-label="Upload CSV or Excel file"
            />
          </label>
        </div>
      </header>

      {error && (
        <div role="alert" className="flex items-start gap-2.5 rounded-xl border border-[var(--red)]/30 bg-[var(--red-soft)] text-[var(--red)] px-4 py-3 text-sm">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {!training && (
        <div
          role="status"
          aria-live="polite"
          className="flex items-center gap-2 rounded-xl border border-[var(--line)] bg-[var(--surface)] px-4 py-2.5 text-xs text-[var(--muted)]"
        >
          <span className={`w-2 h-2 rounded-full shrink-0 ${busy ? 'bg-[var(--amber)] animate-pulse' : 'bg-[var(--green)]'}`} />
          <span className="truncate">{status}</span>
        </div>
      )}

      {(() => {
        return showAdvanced ? (
      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.55fr)] gap-5 items-start">
        <div className="space-y-5">
          <Card>
            <div className="flex items-center justify-between gap-3 mb-4">
              <h2 className="text-sm font-bold flex items-center gap-2">
                <Database className="w-4 h-4 text-[var(--accent)]" /> Dataset & schema
              </h2>
              {dataset && <Badge variant="success" size="sm">{dataset.rows.length} rows</Badge>}
            </div>

            {!dataset ? (
              <div className="rounded-lg border border-dashed border-[var(--line)] px-4 py-8 text-center text-sm text-[var(--muted)]">
                No dataset loaded. Upload a file or use the demo data.
              </div>
            ) : (
              <div className="space-y-4">
                {upload && upload.names.length > 1 && (
                  <div>
                    <label className="block text-xs font-semibold text-[var(--muted)] mb-1.5">
                      Worksheet ({upload.name})
                    </label>
                    <select
                      aria-label="Worksheet"
                      value={sheetName}
                      onChange={(event) => handleSheetChange(event.target.value)}
                      className="w-full px-3 py-2 text-sm bg-[var(--surface)] border border-[var(--line)] rounded-lg"
                    >
                      {upload.names.map((name) => (
                        <option key={name} value={name}>{name}</option>
                      ))}
                    </select>
                  </div>
                )}

                <div className="flex flex-wrap gap-1.5">
                  {profile.idLike?.map((column) => (
                    <Badge key={column} size="sm" variant="warning">ID: {column}</Badge>
                  ))}
                  {profile.targetSource && (
                    <Badge size="sm" variant="neutral">Excluded from training: {profile.targetSource}</Badge>
                  )}
                  {profile.targetBinnedFrom && (
                    <Badge size="sm" variant="accent">Binned from {profile.targetBinnedFrom}</Badge>
                  )}
                </div>

                <div className="max-h-56 overflow-auto rounded-lg border border-[var(--line)]">
                  <table className="w-full text-left text-[11px]">
                    <thead className="sticky top-0 bg-[var(--canvas)] text-[var(--muted)]">
                      <tr>
                        <th className="px-2.5 py-2 font-semibold">Column</th>
                        <th className="px-2.5 py-2 font-semibold">Unique</th>
                        <th className="px-2.5 py-2 font-semibold">Type</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(profile.P ?? {}).map(([column, info]) => (
                        <tr key={column} className="border-t border-[var(--line)]">
                          <td className="px-2.5 py-1.5 font-medium">{pretty(column)}</td>
                          <td className="px-2.5 py-1.5 font-mono">{info.uniques}</td>
                          <td className="px-2.5 py-1.5">
                            {info.isDate ? 'date' : info.numeric ? 'numeric' : 'categorical/text'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </Card>

          <Card>
            <h2 className="text-sm font-bold mb-4 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-[var(--accent)]" /> Training controls
            </h2>
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-semibold text-[var(--muted)] mb-1.5">Target column</label>
                <select
                  aria-label="Target column"
                  value={target}
                  onChange={(event) => setTarget(event.target.value)}
                  disabled={!dataset || training}
                  className="w-full px-3 py-2 text-sm bg-[var(--surface)] border border-[var(--line)] rounded-lg"
                >
                  {columns.map((column) => (
                    <option key={column} value={column}>{pretty(column)}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-[var(--muted)] mb-1.5">Text field for tokenizer</label>
                <select
                  aria-label="Text field for tokenizer"
                  value={textCol}
                  onChange={(event) => setTextCol(event.target.value)}
                  disabled={!dataset || training}
                  className="w-full px-3 py-2 text-sm bg-[var(--surface)] border border-[var(--line)] rounded-lg"
                >
                  {columns.map((column) => (
                    <option key={column} value={column}>{pretty(column)}</option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-[var(--muted)] mb-1.5">Date split</label>
                  <select
                    aria-label="Date split column"
                    value={dateCol}
                    onChange={(event) => setDateCol(event.target.value)}
                    disabled={!dataset || training || split !== 'chron'}
                    className="w-full px-3 py-2 text-sm bg-[var(--surface)] border border-[var(--line)] rounded-lg"
                  >
                    <option value="">Auto detect</option>
                    {dateOptions.map((column) => (
                      <option key={column} value={column}>{pretty(column)}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-[var(--muted)] mb-1.5">Split strategy</label>
                  <select
                    aria-label="Split strategy"
                    value={split}
                    onChange={(event) => setSplit(event.target.value as SplitChoice)}
                    disabled={!dataset || training}
                    className="w-full px-3 py-2 text-sm bg-[var(--surface)] border border-[var(--line)] rounded-lg"
                  >
                    <option value="auto">Auto (random)</option>
                    <option value="random">Random</option>
                    <option value="chron">Chronological</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-[var(--muted)] mb-1.5">GPT tokenizer</label>
                <select
                  aria-label="GPT tokenizer"
                  value={tokenizer}
                  onChange={(event) => setTokenizer(event.target.value as TokenizerChoice)}
                  disabled={training}
                  className="w-full px-3 py-2 text-sm bg-[var(--surface)] border border-[var(--line)] rounded-lg"
                >
                  <option value="cl100k">Hugging Face · cl100k (GPT-3.5/4)</option>
                  <option value="o200k">Hugging Face · o200k (GPT-4o)</option>
                  <option value="builtin">Built-in byte-level BPE (offline)</option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <label className="block text-xs font-semibold text-[var(--muted)]">
                  Token vocab: {vocab}
                  <input
                    type="range"
                    min={512}
                    max={8192}
                    step={512}
                    value={vocab}
                    onChange={(event) => setVocab(Number(event.target.value))}
                    disabled={training}
                    className="w-full mt-1.5 accent-[var(--accent)]"
                  />
                </label>
                <div className="block text-xs font-semibold text-[var(--muted)]">
                  Deep model epochs: {epochs} (fixed)
                  <div className="w-full mt-1.5 h-1.5 rounded-full bg-[var(--canvas)] border border-[var(--line)]" />
                </div>
              </div>

              <Button variant="primary" className="w-full" onClick={handleTrain} disabled={!dataset || busy || training}>
                {training ? `Training ${progress.done}/${progress.total}…` : 'Train 10 deep models'}
              </Button>

              {training && (
                <div>
                  <div className="h-2 rounded-full bg-[var(--canvas)] overflow-hidden border border-[var(--line)]">
                    <div className="h-full bg-[var(--accent)] transition-all" style={{ width: `${progressPercent}%` }} />
                  </div>
                  <p className="text-[11px] text-[var(--muted)] mt-1.5 truncate">{progress.label}</p>
                </div>
              )}
            </div>
          </Card>
        </div>

        {renderMain()}
      </div>
      ) : (
        renderMain()
      );

        function renderMain() {
          return (
        <div className="space-y-5">
          {training && <TrainingProgress progress={progressPercent} detail={showAdvanced ? progress.label : undefined} />}

          {!result && !training ? (
            <Card>
              <div className="rounded-lg border border-dashed border-[var(--line)] px-6 py-12 text-center">
                <Sparkles className="w-7 h-7 text-[var(--faint)] mx-auto mb-3" />
                <h2 className="text-sm font-bold mb-1">No file yet — knowledge is ready</h2>
                <p className="text-xs text-[var(--muted)]">
                  You can ask about any topic right now using ViDo’s built-in knowledge. Upload a CSV or Excel file any time to add your own data.
                </p>
              </div>
            </Card>
          ) : (
            <>
              {result && (
              <>
              <Card>
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <Badge variant="success">Ready</Badge>
                      {showAdvanced && result.binned && <Badge variant="accent" size="sm">Target auto-binned</Badge>}
                      {showAdvanced && result.capped && <Badge variant="warning" size="sm">Capped at 1,200 rows</Badge>}
                    </div>
                    <h2 className="text-lg font-bold">Your models are trained</h2>
                    <p className="text-xs text-[var(--muted)] mt-1">
                      Learned from {result.dataset.rows.length} rows and saved for this account.
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-black text-[var(--green)]">{metricText(result.best.acc)}</div>
                    <div className="text-[11px] text-[var(--muted)]">
                      {datasetLoaded ? 'accuracy on test rows' : 'upload data to see accuracy'}
                    </div>
                  </div>
                </div>

                {datasetLoaded && (
                  <div className="mt-4">
                    <div className="text-xs font-semibold mb-2">All model scores</div>
                    <div className="max-h-48 overflow-auto pr-1 space-y-1.5">
                      {result.metrics.map((metric) => (
                        <div
                          key={metric.n}
                          className={`flex items-center justify-between gap-2 rounded-lg border px-2.5 py-1.5 text-[11px] ${
                            metric.n === result.best.n
                              ? 'border-[var(--green)] bg-[var(--green-soft)]'
                              : 'border-[var(--line)] bg-[var(--canvas)]'
                          }`}
                        >
                          <span className="font-medium truncate">{metric.err ? '⚠ ' : ''}{metric.n}</span>
                          <span className="font-mono whitespace-nowrap">
                            {metric.err ? 'failed' : `${metricText(metric.acc)} · F1 macro ${metricText(metric.F)}`}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {showAdvanced && (
                  <div className="mt-4 space-y-3">
                    <div className="text-xs font-semibold">
                      {result.best.n} · {result.metrics.length} deep models compared
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                      <div className="rounded-lg bg-[var(--canvas)] border border-[var(--line)] p-2.5">
                        <div className="text-[var(--faint)]">Precision</div>
                        <div className="font-bold">{metricText(result.best.P)}</div>
                      </div>
                      <div className="rounded-lg bg-[var(--canvas)] border border-[var(--line)] p-2.5">
                        <div className="text-[var(--faint)]">Recall</div>
                        <div className="font-bold">{metricText(result.best.R)}</div>
                      </div>
                      <div className="rounded-lg bg-[var(--canvas)] border border-[var(--line)] p-2.5">
                        <div className="text-[var(--faint)]">F1 macro</div>
                        <div className="font-bold">{metricText(result.best.F)}</div>
                      </div>
                      <div className="rounded-lg bg-[var(--canvas)] border border-[var(--line)] p-2.5">
                        <div className="text-[var(--faint)]">Split</div>
                        <div className="font-bold">{result.trainSize}/{result.testSize}</div>
                      </div>
                    </div>
                    <p className="text-[11px] text-[var(--muted)]">{result.splitNote}</p>
                    <div className="max-h-72 overflow-auto rounded-lg border border-[var(--line)]">
                      <table className="w-full text-left text-[11px]">
                        <thead className="sticky top-0 bg-[var(--canvas)] text-[var(--muted)]">
                          <tr>
                            <th className="px-2.5 py-2 font-semibold">Model</th>
                            <th className="px-2.5 py-2 font-semibold">Accuracy</th>
                            <th className="px-2.5 py-2 font-semibold">F1 macro</th>
                            <th className="px-2.5 py-2 font-semibold">Time</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.metrics.map((metric) => (
                            <tr
                              key={metric.n}
                              className={`border-t border-[var(--line)] ${metric.n === result.best.n ? 'bg-[var(--green-soft)]' : ''}`}
                            >
                              <td className="px-2.5 py-1.5 font-medium">{metric.err ? '⚠ ' : ''}{metric.n}</td>
                              <td className="px-2.5 py-1.5 font-mono">{metric.err ? 'failed' : metricText(metric.acc)}</td>
                              <td className="px-2.5 py-1.5 font-mono">{metric.err ? '—' : metricText(metric.F)}</td>
                              <td className="px-2.5 py-1.5 font-mono">{metric.ms} ms</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </Card>

              <Card>
                <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
                  <div>
                    <h2 className="text-sm font-bold">Model probability</h2>
                    <p className="text-xs text-[var(--muted)] mt-1">
                      Pick any trained model and a deadline to see its chance of finishing on time.
                    </p>
                  </div>
                  <Badge size="sm" variant="neutral">{selectedArtifact?.name ?? result.best.n}</Badge>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                  <label className="block text-xs font-semibold text-[var(--muted)]">
                    Model
                    <select
                      aria-label="Model"
                      value={selectedModel}
                      onChange={(event) => setSelectedModel(event.target.value)}
                      className="w-full mt-1.5 rounded-lg border border-[var(--line)] bg-[var(--canvas)] px-2.5 py-2 text-xs"
                    >
                      {(selectableMetrics.length ? selectableMetrics : result.metrics).map((metric) => (
                        <option key={metric.n} value={metric.n}>{metric.n}</option>
                      ))}
                    </select>
                  </label>
                  <label className="block text-xs font-semibold text-[var(--muted)]">
                    Task
                    <input
                      aria-label="Task for deadline probability"
                      value={deadlineTask}
                      onChange={(event) => setDeadlineTask(event.target.value)}
                      placeholder="Describe the task"
                      className="w-full mt-1.5 rounded-lg border border-[var(--line)] bg-[var(--canvas)] px-2.5 py-2 text-xs"
                    />
                  </label>
                  <label className="block text-xs font-semibold text-[var(--muted)]">
                    Deadline
                    <input
                      type="date"
                      aria-label="Deadline"
                      value={deadlineDate}
                      onChange={(event) => setDeadlineDate(event.target.value)}
                      className="w-full mt-1.5 rounded-lg border border-[var(--line)] bg-[var(--canvas)] px-2.5 py-2 text-xs"
                    />
                  </label>
                </div>
                <div className="mt-3 flex flex-wrap items-end justify-between gap-3 rounded-lg bg-[var(--canvas)] border border-[var(--line)] p-3">
                  <div>
                    <div className="text-[11px] text-[var(--muted)]">
                      Chance of completing within {deadlineDate || 'the selected deadline'}
                    </div>
                    <div className="text-[11px] text-[var(--faint)] mt-0.5">
                      {datasetLoaded
                        ? `Using ${selectedArtifact?.name ?? result.best.n}`
                        : 'Load a dataset to see estimates'}
                    </div>
                  </div>
                  <div data-testid="deadline-probability" className="text-2xl font-black text-[var(--green)]">
                    {deadlineProbability === null ? '—' : formatConfidence(deadlineProbability)}
                  </div>
                </div>
                {datasetLoaded && completionLabelIndex < 0 && (
                  <p className="text-[11px] text-[var(--amber)] mt-2">
                    This dataset has no completed-on-time label to estimate.
                  </p>
                )}
              </Card>

              {showAdvanced && (
                <Card>
                  <h2 className="text-sm font-bold mb-3">Tokenizer verification</h2>
                  <p className="text-xs text-[var(--muted)] mb-3">{result.tokenizerNote}</p>
                  {result.tokenizerWarning && (
                    <p className="text-[11px] text-[var(--amber)] mb-3">{result.tokenizerWarning}</p>
                  )}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div className="rounded-lg border border-[var(--line)] p-3 text-[11px] space-y-2">
                      <div><span className="font-semibold">Example:</span> {result.tokenizer.example.slice(0, 160) || '—'}</div>
                      <div className="font-mono break-all"><span className="font-sans font-semibold">Raw IDs:</span> {result.tokenizer.raw.join(' ') || '—'}</div>
                      <div className="font-mono break-all"><span className="font-sans font-semibold">Compact IDs:</span> {result.tokenizer.compact.join(' ') || '—'}</div>
                      <div className="break-words"><span className="font-semibold">Subwords:</span> {result.tokenizer.subwords || '—'}</div>
                      <div className="break-words"><span className="font-semibold">Decoded:</span> {result.tokenizer.decoded || '—'}</div>
                    </div>
                    <div className="space-y-2">
                      <div className="rounded-lg border border-[var(--line)] p-2.5">
                        <div className="text-[11px] font-semibold mb-1">Training loss</div>
                        {datasetLoaded ? (
                          <svg viewBox="0 0 320 120" className="w-full h-20" role="img" aria-label="Training loss chart">
                            <polyline points={sparkline(result.history.lt, lossMax)} fill="none" stroke="var(--accent)" strokeWidth="2" />
                            <polyline points={sparkline(result.history.le, lossMax)} fill="none" stroke="var(--amber)" strokeWidth="2" />
                          </svg>
                        ) : (
                          <p className="text-[11px] text-[var(--muted)]">Load a dataset to see training progress.</p>
                        )}
                      </div>
                      <div className="rounded-lg border border-[var(--line)] p-2.5">
                        <div className="text-[11px] font-semibold mb-1">Training accuracy</div>
                        {datasetLoaded ? (
                          <svg viewBox="0 0 320 120" className="w-full h-20" role="img" aria-label="Training accuracy chart">
                            <polyline points={sparkline(result.history.at, accuracyMax)} fill="none" stroke="var(--accent)" strokeWidth="2" />
                            <polyline points={sparkline(result.history.ae, accuracyMax)} fill="none" stroke="var(--green)" strokeWidth="2" />
                          </svg>
                        ) : (
                          <p className="text-[11px] text-[var(--muted)]">Load a dataset to see training progress.</p>
                        )}
                      </div>
                    </div>
                  </div>
                </Card>
              )}

              {showAdvanced && (
                <>
              <Card>
                <div className="flex items-center justify-between gap-2 mb-3">
                  <h2 className="text-sm font-bold flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-[var(--accent)]" /> Suggestions
                  </h2>
                  <Badge size="sm" variant="neutral">{result.suggestions.length} generated</Badge>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                  {result.suggestions.map((suggestion) => (
                    <div key={suggestion.id} className="rounded-lg border border-[var(--line)] p-3">
                      <div className="flex items-start justify-between gap-2">
                        <div className="text-xs font-semibold line-clamp-2">{suggestion.title}</div>
                        <Badge size="sm" variant="success">{suggestion.label}</Badge>
                      </div>
                      <div className="text-[11px] text-[var(--muted)] mt-1.5">
                        {datasetLoaded ? `${suggestion.conf} chance` : 'Load a dataset to see estimates'}
                      </div>
                      {showAdvanced && suggestion.fields && (
                        <div className="text-[10px] text-[var(--faint)] mt-1 line-clamp-1">{suggestion.fields}</div>
                      )}
                    </div>
                  ))}
                </div>
              </Card>

              <Card>
                <h2 className="text-sm font-bold mb-3">Try a prediction</h2>
                <div className="space-y-3">
                  {predictionFields.map((column) =>
                    column === result.textCol ? (
                      <div key={column}>
                        <label className="block text-xs font-semibold text-[var(--muted)] mb-1.5">{pretty(column)}</label>
                        <textarea
                          rows={3}
                          value={predictionValues[column] ?? ''}
                          onChange={(event) =>
                            setPredictionValues((prev) => ({ ...prev, [column]: event.target.value }))
                          }
                          className="w-full px-3.5 py-2 text-sm bg-[var(--surface)] border border-[var(--line)] rounded-lg"
                        />
                      </div>
                    ) : (
                      <Input
                        key={column}
                        label={pretty(column)}
                        value={predictionValues[column] ?? ''}
                        onChange={(event) =>
                          setPredictionValues((prev) => ({ ...prev, [column]: event.target.value }))
                        }
                      />
                    )
                  )}
                  <Button
                    variant="primary"
                    size="sm"
                    onClick={handlePredict}
                    disabled={!predictionFields.length || !modelReady}
                  >
                    {modelReady ? `Predict ${pretty(result.target)}` : 'Preparing saved model…'}
                  </Button>
                  {!modelReady && (
                    <p className="text-[11px] text-[var(--muted)]">
                      The saved model is being rebuilt from your data. Predictions will unlock automatically.
                    </p>
                  )}
                  {prediction && (
                    <div className="flex flex-wrap items-center gap-2 rounded-lg border border-[var(--green)]/30 bg-[var(--green-soft)] px-3 py-2.5">
                      <CheckCircle2 className="w-4 h-4 text-[var(--green)]" />
                      <span className="text-sm font-bold">{prediction.label}</span>
                      {datasetLoaded && <Badge size="sm" variant="success">{prediction.confidence} chance</Badge>}
                    </div>
                  )}
                </div>
              </Card>
                </>
              )}
              </>
            )}
          </>
        )}

        </div>
          );
        }
      })()}
    </div>
  );
};

export default MLLabPage;
