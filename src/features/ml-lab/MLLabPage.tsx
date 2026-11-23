import React, { useEffect, useMemo, useState } from 'react';
import { AlertCircle, BarChart3, CheckCircle2, Database, Search, Sparkles, Upload } from 'lucide-react';
import { Badge, Button, Card, Input } from '../../components/ui';
import {
  demoRows,
  loadUploadedFile,
  pct,
  predictWith,
  prepareDataset,
  profileData,
  ragDocs,
  runRAG,
  selectUploadSheet,
  sparkline,
  syncPredictionRow,
  trainAll,
  USER_RAG_MODE,
} from './engine';
import type {
  Dataset,
  PreparedData,
  Profile,
  RAGDoc,
  RAGResult,
  Row,
  TrainingOptions,
  TrainingResult,
  UploadedData,
} from './engine';
import {
  cancelRestore,
  clearActiveResult,
  getActiveResult,
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
import { polishRagAnswer } from './answerEngine';
import { scheduleLocalModelWarmup } from './languageModel';
import { RagWorking, TrainingProgress } from './WorkingIndicator';

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
  const [sheetName, setSheetName] = useState(initialSnapshot?.sheetName ?? '');
  const [dataset, setDataset] = useState<Dataset | null>(initialSnapshot?.dataset ?? null);
  const [profile, setProfile] = useState<Profile>(() => initialSnapshot?.profile ?? profileData([], []));

  const [target, setTarget] = useState(initialSnapshot?.target ?? '');
  const [textCol, setTextCol] = useState(initialSnapshot?.textCol ?? '');
  const [dateCol, setDateCol] = useState(initialSnapshot?.dateCol ?? '');
  const [tokenizer, setTokenizer] = useState<TokenizerChoice>(initialSnapshot?.tokenizer ?? 'cl100k');
  const [split, setSplit] = useState<SplitChoice>(initialSnapshot?.split ?? 'auto');
  const [vocab, setVocab] = useState(initialSnapshot?.vocab ?? 2048);
  const [epochs, setEpochs] = useState(initialSnapshot?.epochs ?? 6);

  const [result, setResult] = useState<TrainingResult | null>(() =>
    resultFromStored(initialSnapshot?.result ?? null, getActiveResult(userId))
  );
  const [modelReady, setModelReady] = useState(() => Boolean(getActiveResult(userId)));
  const [predictionValues, setPredictionValues] = useState<Row>(initialSnapshot?.predictionValues ?? {});
  const [prediction, setPrediction] = useState<{ label: string; confidence: string; model: string } | null>(null);

  const [ragQuery, setRagQuery] = useState('');
  const [ragResult, setRagResult] = useState<RAGResult | null>(null);
  const [ragNotice, setRagNotice] = useState('');
  const [ragBusy, setRagBusy] = useState(false);
  const [ragStage, setRagStage] = useState('');
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
    setEpochs(snapshot?.epochs ?? 6);
    setResult(resultFromStored(snapshot?.result ?? null, active));
    setModelReady(Boolean(active));
    setPredictionValues(snapshot?.predictionValues ?? {});
    setPrediction(null);
    setRagResult(null);
    setRagNotice('');
    setRagBusy(false);
    setRagStage('');
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
    if (snapshot?.result || snapshot?.dataset) scheduleLocalModelWarmup();
  }, [userId]);

  useEffect(
    () =>
      subscribeActiveResult(userId, (active) => {
        if (!active) return;
        setResult(active);
        setModelReady(true);
        setStatus('Saved model ready.');
      }),
    [userId]
  );

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
    setRagResult(null);
    setRagNotice('');
    setRagBusy(false);
    setRagStage('');
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
      const trained = await trainAll(
        activeDataset,
        activeProfile,
        options,
        () => undefined,
        (done, total, label) => setProgress({ done, total, label })
      );
      setResult(trained);
      setActiveResult(userId, trained);
      setModelReady(true);
      setProfile(trained.profile);
      setTarget(trained.target);
      setTextCol(trained.textCol);
      setStatus('Ready. Ask your data anything.');
      scheduleLocalModelWarmup();
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
    setDataset(data.dataset);
    setProfile(data.profile);
    setTarget(data.profile.target || data.dataset.target || '');
    setTextCol(data.profile.textCol || data.dataset.cols[0] || '');
    setDateCol(data.profile.primaryDate || '');
    setResult(null);
    setModelReady(false);
    setPredictionValues({});
    setPrediction(null);
    setRagResult(null);
    setRagNotice('');
    setRagBusy(false);
    setRagStage('');
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

  const docs = useMemo<RAGDoc[]>(
    () => ragDocs(result?.dataset ?? dataset, result?.profile ?? profile, result?.suggestions ?? []),
    [result, dataset, profile]
  );

  const ragExamples = useMemo(() => {
    const t = target || profile.target || 'target';
    const txt = textCol || profile.textCol || 'text';
    const category = profile.cats?.[0] || 'category';
    return [
      `Which ${txt} records look likely to be ${t}?`,
      `Compare ${category} and ${t} across the dataset`,
      `What patterns predict ${t}?`,
    ];
  }, [target, textCol, profile]);

  const handlePredict = () => {
    if (!result) return;
    setPrediction(predictWith(result, syncPredictionRow(result, predictionValues)));
  };

  const handleRAG = async () => {
    if (!docs.length) {
      setError('Upload a file before asking a question.');
      return;
    }
    if (!ragQuery.trim()) {
      setRagResult(null);
      setRagNotice('Type a question or pick an example below to search your data.');
      return;
    }
    const question = ragQuery.trim();
    setError('');
    setRagNotice('');
    setRagBusy(true);
    setRagStage('');
    setRagResult(null);
    const startedAt = Date.now();
    await new Promise<void>((resolve) => window.setTimeout(resolve, 30));
    try {
      const retrieved = runRAG(question, USER_RAG_MODE, docs, result?.profile ?? profile, result?.suggestions ?? []);
      setRagStage('Comparing the best matches…');
      const next = await polishRagAnswer({ question, rag: retrieved, onProgress: setRagStage });
      const remaining = 420 - (Date.now() - startedAt);
      if (remaining > 0) {
        await new Promise<void>((resolve) => window.setTimeout(resolve, remaining));
      }
      setRagResult(next);
    } finally {
      setRagBusy(false);
      setRagStage('');
    }
  };

  const progressPercent = Math.min(100, Math.round((progress.done / Math.max(1, progress.total)) * 100));
  const answerLines = useMemo(() => {
    if (!ragResult?.answer) return [];
    const parts = ragResult.answer.split(/\.\s+/).map((part) => part.trim()).filter(Boolean);
    return parts.map((part, index) => (index < parts.length - 1 ? `${part}.` : part));
  }, [ragResult]);
  const savedLabel = savedAt
    ? new Date(savedAt).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
    : '';
  const lossMax = result ? Math.max(0.1, ...result.history.lt, ...result.history.le) : 1;
  const accuracyMax = result ? Math.max(0.1, ...result.history.at, ...result.history.ae) : 1;

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-[1500px] mx-auto space-y-5">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Badge variant="accent" size="sm">Simple mode</Badge>
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
          <span className={`w-2 h-2 rounded-full shrink-0 ${busy || ragBusy ? 'bg-[var(--amber)] animate-pulse' : 'bg-[var(--green)]'}`} />
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

                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="rounded-lg bg-[var(--canvas)] border border-[var(--line)] p-2.5">
                    <div className="text-[var(--faint)]">Columns</div>
                    <div className="font-bold text-sm">{dataset.cols.length}</div>
                  </div>
                  <div className="rounded-lg bg-[var(--canvas)] border border-[var(--line)] p-2.5">
                    <div className="text-[var(--faint)]">Categorical</div>
                    <div className="font-bold text-sm">{profile.cats?.length ?? 0}</div>
                  </div>
                  <div className="rounded-lg bg-[var(--canvas)] border border-[var(--line)] p-2.5">
                    <div className="text-[var(--faint)]">Text fields</div>
                    <div className="font-bold text-sm">{profile.texts?.length ?? 0}</div>
                  </div>
                  <div className="rounded-lg bg-[var(--canvas)] border border-[var(--line)] p-2.5">
                    <div className="text-[var(--faint)]">Date fields</div>
                    <div className="font-bold text-sm">{profile.dates?.length ?? 0}</div>
                  </div>
                </div>

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
                <label className="block text-xs font-semibold text-[var(--muted)]">
                  LSTM epochs: {epochs}
                  <input
                    type="range"
                    min={2}
                    max={20}
                    step={1}
                    value={epochs}
                    onChange={(event) => setEpochs(Number(event.target.value))}
                    disabled={training}
                    className="w-full mt-1.5 accent-[var(--accent)]"
                  />
                </label>
              </div>

              <Button variant="primary" className="w-full" onClick={handleTrain} disabled={!dataset || busy || training}>
                {training ? `Training ${progress.done}/${progress.total}…` : 'Train 10 models'}
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
                <h2 className="text-sm font-bold mb-1">Nothing learned yet</h2>
                <p className="text-xs text-[var(--muted)]">
                  Upload a CSV or Excel file. ViDo will work out the rest.
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
                    <h2 className="text-lg font-bold">Ready to answer questions</h2>
                    <p className="text-xs text-[var(--muted)] mt-1">
                      Learned from {result.dataset.rows.length} rows and saved for this account.
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-black text-[var(--green)]">{pct(result.best.acc)}</div>
                    <div className="text-[11px] text-[var(--muted)]">accuracy on test rows</div>
                  </div>
                </div>

                {showAdvanced && (
                  <div className="mt-4 space-y-3">
                    <div className="text-xs font-semibold">
                      {result.best.n} · {result.metrics.length} models compared
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                      <div className="rounded-lg bg-[var(--canvas)] border border-[var(--line)] p-2.5">
                        <div className="text-[var(--faint)]">Precision</div>
                        <div className="font-bold">{pct(result.best.P)}</div>
                      </div>
                      <div className="rounded-lg bg-[var(--canvas)] border border-[var(--line)] p-2.5">
                        <div className="text-[var(--faint)]">Recall</div>
                        <div className="font-bold">{pct(result.best.R)}</div>
                      </div>
                      <div className="rounded-lg bg-[var(--canvas)] border border-[var(--line)] p-2.5">
                        <div className="text-[var(--faint)]">F1 score</div>
                        <div className="font-bold">{pct(result.best.F)}</div>
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
                            <th className="px-2.5 py-2 font-semibold">F1</th>
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
                              <td className="px-2.5 py-1.5 font-mono">{metric.err ? 'failed' : pct(metric.acc)}</td>
                              <td className="px-2.5 py-1.5 font-mono">{metric.err ? '—' : pct(metric.F)}</td>
                              <td className="px-2.5 py-1.5 font-mono">{metric.ms} ms</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
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
                        <div className="text-[11px] font-semibold mb-1">LSTM loss</div>
                        <svg viewBox="0 0 320 120" className="w-full h-20" role="img" aria-label="LSTM loss chart">
                          <polyline points={sparkline(result.history.lt, lossMax)} fill="none" stroke="var(--accent)" strokeWidth="2" />
                          <polyline points={sparkline(result.history.le, lossMax)} fill="none" stroke="var(--amber)" strokeWidth="2" />
                        </svg>
                      </div>
                      <div className="rounded-lg border border-[var(--line)] p-2.5">
                        <div className="text-[11px] font-semibold mb-1">LSTM accuracy</div>
                        <svg viewBox="0 0 320 120" className="w-full h-20" role="img" aria-label="LSTM accuracy chart">
                          <polyline points={sparkline(result.history.at, accuracyMax)} fill="none" stroke="var(--accent)" strokeWidth="2" />
                          <polyline points={sparkline(result.history.ae, accuracyMax)} fill="none" stroke="var(--green)" strokeWidth="2" />
                        </svg>
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
                        {suggestion.conf} chance
                      </div>
                      {showAdvanced && suggestion.fields && (
                        <div className="text-[10px] text-[var(--faint)] mt-1 line-clamp-1">{suggestion.fields}</div>
                      )}
                      <button
                        type="button"
                        onClick={() => setRagQuery(`Tell me about ${suggestion.title} and why it is ${suggestion.label}`)}
                        className="text-[11px] font-semibold text-[var(--accent)] hover:underline mt-2 cursor-pointer"
                      >
                        Ask about this →
                      </button>
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
                      <Badge size="sm" variant="success">{prediction.confidence} chance</Badge>
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

          <Card className="border-[var(--accent)]/20">
            <div className="flex items-start justify-between gap-3 mb-3">
              <div>
                <h2 className="text-sm font-bold flex items-center gap-2">
                  <Search className="w-4 h-4 text-[var(--accent)]" /> Ask your data
                </h2>
                <p className="text-[11px] text-[var(--muted)] mt-1">
                  Ask anything about{' '}
                  <strong className="text-[var(--ink)]">
                    {upload ? `${upload.name}${sheetName ? ` · ${sheetName}` : ''}` : 'your saved file'}
                  </strong>
                  . The file was detected and saved automatically.
                </p>
              </div>
              {showAdvanced && <Badge size="sm" variant="neutral">{docs.length} sources</Badge>}
            </div>

            {showAdvanced && (
              <p className="text-[10px] text-[var(--faint)] mb-3">
                ViDo always uses agentic retrieval — it breaks your question into steps when needed and writes the final answer with the local assistant.
              </p>
            )}

            <textarea
              rows={3}
              value={ragQuery}
              onChange={(event) => {
                setRagQuery(event.target.value);
                setRagNotice('');
              }}
              placeholder="Ask anything about your file…"
              className="w-full px-3.5 py-2 text-sm bg-[var(--surface)] border border-[var(--line)] rounded-lg"
            />
            <div className="flex flex-wrap gap-1.5 mt-2">
              {ragExamples.map((example) => (
                <button
                  key={example}
                  type="button"
                  onClick={() => {
                    setRagQuery(example);
                    setRagNotice('');
                  }}
                  className="text-[10px] px-2 py-1 rounded-full border border-[var(--line)] text-[var(--muted)] hover:bg-[var(--canvas)] cursor-pointer"
                >
                  {example}
                </button>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-3 mt-3">
              <Button variant="secondary" size="sm" onClick={handleRAG} disabled={!docs.length || ragBusy}>
                {ragBusy ? 'Finding…' : 'Find answer'}
              </Button>
              {ragNotice && <span className="text-[11px] text-[var(--muted)]">{ragNotice}</span>}
            </div>

            {ragBusy && <RagWorking stage={ragStage} />}

            {ragResult && !ragBusy && (
              <div className="mt-4 space-y-3">
                {showAdvanced && (
                  <div className="flex flex-wrap items-center gap-2 rounded-lg bg-[var(--accent-soft)] px-3 py-2">
                    <Badge size="sm" variant="accent">Agentic mode</Badge>
                    <span className="text-[11px] text-[var(--muted)]">
                      {ragResult.why.charAt(0).toLowerCase() + ragResult.why.slice(1)}
                    </span>
                  </div>
                )}

                <div className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Sparkles className="w-3.5 h-3.5 text-[var(--accent)]" />
                    <div className="text-[11px] font-bold uppercase tracking-wide text-[var(--muted)]">Answer</div>
                  </div>
                  {answerLines.length ? (
                    <ul className="space-y-1.5 text-sm text-[var(--ink)]">
                      {answerLines.map((line, index) => (
                        <li key={`${line}-${index}`} className="flex gap-2">
                          <span className="text-[var(--accent)] mt-1.5 text-[8px]">●</span>
                          <span>{line}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm text-[var(--muted)]">{ragResult.answer}</p>
                  )}
                </div>

                {showAdvanced && (ragResult.plan.length > 0 || ragResult.trail) && (
                  <details className="rounded-lg border border-[var(--line)] px-3 py-2 text-[11px] text-[var(--muted)]">
                    <summary className="cursor-pointer font-semibold text-[var(--ink)]">
                      How this answer was found
                    </summary>
                    {ragResult.plan.length > 0 && (
                      <ol className="list-decimal list-inside mt-2 space-y-1">
                        {ragResult.plan.map((step, index) => (
                          <li key={`${step}-${index}`}>{step}</li>
                        ))}
                      </ol>
                    )}
                    {ragResult.trail && <p className="mt-2 text-[var(--faint)]">{ragResult.trail}</p>}
                  </details>
                )}

                {showAdvanced && ragResult.ranked.length > 0 && (
                  <div>
                    <div className="text-[11px] font-bold text-[var(--muted)] mb-2">Most relevant sources</div>
                    <div className="space-y-1.5">
                      {ragResult.ranked.map((item, index) => (
                        <div key={`${item.title}-${index}`} className="rounded-lg border border-[var(--line)] p-2.5">
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-xs font-semibold truncate">{item.title}</span>
                            <span className="text-[10px] font-semibold text-[var(--muted)] whitespace-nowrap">
                              Match score · {item.score.toFixed(1)}
                            </span>
                          </div>
                          <div className="text-[11px] text-[var(--muted)] mt-1 line-clamp-2">{item.excerpt}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </Card>
        </div>
          );
        }
      })()}
    </div>
  );
};

export default MLLabPage;
