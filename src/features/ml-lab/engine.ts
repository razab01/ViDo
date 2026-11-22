/**
 * Dynamic ML/RAG engine.
 *
 * This is the user-added feature layer. Everything here is dataset-driven:
 * CSV/TSV/XLSX/XLS/ODS ingestion, schema inference, dynamic targets and text
 * fields, 10 in-browser classifiers, real Hugging Face GPT tokenizers with a
 * byte-level BPE fallback, best-model suggestions, and schema-aware RAG.
 */

export type Row = Record<string, string>;

export interface Dataset {
  cols: string[];
  rows: Row[];
  target: string;
}

export interface ColumnProfile {
  n: number;
  uniques: number;
  ratio: number;
  numeric: boolean;
  isDate: boolean;
}

export interface Profile {
  P: Record<string, ColumnProfile>;
  idLike: string[];
  dates: string[];
  texts: string[];
  cats: string[];
  outcome: string | null;
  textCol: string | null;
  primaryDate: string | null;
  target: string;
  targetSource?: string;
  targetBinnedFrom?: string;
  rows?: Row[];
  allCols?: string[];
}

export interface PreparedData {
  dataset: Dataset;
  profile: Profile;
  valid: boolean;
  message: string;
}

export interface UploadedData {
  name: string;
  names: string[];
  sheets: Record<string, { cols: string[]; rows: Row[] }>;
  buffer?: ArrayBuffer;
  sheet: string;
  data: PreparedData;
}

export interface Suggestion {
  id: string | number;
  title: string;
  label: string;
  conf: string;
  model: string;
  target: string;
  fields: string;
}

export interface ModelMetric {
  n: string;
  acc: number;
  P: number;
  R: number;
  F: number;
  ms: number;
  err?: boolean;
}

export interface Artifact {
  name: string;
  predict: (row: Row) => number;
  proba?: (row: Row) => number[];
}

export interface TrainingOptions {
  target: string;
  textCol: string;
  tokenizer: 'cl100k' | 'o200k' | 'builtin';
  split: 'auto' | 'random' | 'chron';
  vocab: number;
  epochs: number;
  bpeEmb: number;
  bpeHidden: number;
  bpeEpochs: number;
}

export interface TrainingResult {
  dataset: Dataset;
  profile: Profile;
  target: string;
  textCol: string;
  labels: string[];
  features: string[];
  splitNote: string;
  capped: boolean;
  trainSize: number;
  testSize: number;
  tokenizerNote: string;
  tokenizerWarning: string;
  tokenizer: {
    example: string;
    raw: number[];
    compact: number[];
    subwords: string;
    decoded: string;
  };
  history: { lt: number[]; le: number[]; at: number[]; ae: number[] };
  metrics: ModelMetric[];
  best: ModelMetric;
  artifact: Artifact;
  suggestions: Suggestion[];
  sample: Row;
  baseline: number;
  binned: boolean;
}

export interface RAGDoc {
  id: string | number;
  title: string;
  text: string;
  tags: string;
  priority: string;
  assignee: string;
  edges: unknown[];
  suggestion?: boolean;
  label?: string;
  conf?: string;
  model?: string;
  target?: string;
  sourceTitle?: string;
  ents?: string[];
}

export interface RAGResult {
  mode: string;
  why: string;
  plan: string[];
  trail: string;
  ranked: Array<{ title: string; score: number; excerpt: string }>;
  answer: string;
}

type Matrix = number[][];
type Store = { predict?: (x: number[]) => number; proba?: (x: number[]) => number[] };
type ModelFn = (X: Matrix, y: number[], Xt: Matrix, K: number, store: Store) => number[];

interface Tree {
  leaf?: number;
  f?: number;
  th?: number;
  l?: Tree;
  r?: Tree;
}

interface TextEncoderLike {
  kind: 'hf' | 'builtin';
  size: number;
  topCount: number;
  raw: (text: string) => number[];
  encode: (text: string) => number[];
  show: (ids: number[]) => string;
  decode: (ids: number[]) => string;
}

interface LSTMModel {
  V: number;
  E: number;
  H: number;
  K: number;
  forward: (seq: number[], cache?: boolean) => Float32Array | { prob: Float32Array; hs: Float32Array[]; cs: Float32Array[]; zs: Float32Array[] };
  train: (seq: number[], y: number, lr: number) => number;
}

declare global {
  interface Window {
    XLSX?: any;
  }
}

/* ------------------------------------------------------------------ */
/* Deterministic helpers                                               */
/* ------------------------------------------------------------------ */

export function mulberry(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function shuffleIdx(n: number, seed: number): number[] {
  const a = Array.from({ length: n }, (_, i) => i);
  const r = mulberry(seed);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    const tmp = a[i];
    a[i] = a[j];
    a[j] = tmp;
  }
  return a;
}

export function pct(v: number): string {
  return (Number(v) * 100).toFixed(1) + '%';
}

export function fmtNum(v: string | number): string {
  const n = Number(v);
  if (!Number.isFinite(n)) return String(v);
  return Math.abs(n) >= 100 ? n.toFixed(0) : Math.abs(n) >= 10 ? n.toFixed(1) : n.toFixed(2);
}

export function argmax(a: ArrayLike<number>): number {
  let b = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i;
  return b;
}

/* ------------------------------------------------------------------ */
/* CSV / Excel ingestion and schema inference                          */
/* ------------------------------------------------------------------ */

export function parseDelimited(s: string, d: string): string[][] {
  const rows: string[][] = [];
  let r: string[] = [];
  let c = '';
  let inq = false;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (inq) {
      if (ch === '"') {
        if (s[i + 1] === '"') {
          c += '"';
          i++;
        } else inq = false;
      } else c += ch;
    } else if (ch === '"') inq = true;
    else if (ch === d) {
      r.push(c);
      c = '';
    } else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && s[i + 1] === '\n') i++;
      r.push(c);
      c = '';
      if (r.some((x) => String(x).trim() !== '')) rows.push(r);
      r = [];
    } else c += ch;
  }
  r.push(c);
  if (r.some((x) => String(x).trim() !== '')) rows.push(r);
  return rows;
}

export function parseCSV(s: string): string[][] {
  const h = s.split('\n')[0] || '';
  const d = h.includes('\t') ? '\t' : h.includes(';') && !h.includes(',') ? ';' : ',';
  return parseDelimited(s, d);
}

export function cellText(v: unknown): string {
  if (v == null) return '';
  if (v instanceof Date) return isNaN(v.getTime()) ? '' : v.toISOString().slice(0, 10);
  if (typeof v === 'number') return Number.isFinite(v) ? String(v) : '';
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  return String(v).trim();
}

export function excelSerialToISO(v: unknown): string {
  const s = String(v ?? '').trim();
  if (!/^\d{5}(?:\.\d+)?$/.test(s)) return s;
  const n = +s;
  if (n < 20000 || n > 80000) return s;
  const d = new Date(Math.round((n - 25569) * 86400000));
  return isNaN(d.getTime()) ? s : d.toISOString().slice(0, 10);
}

export function dateNum(v: unknown): number {
  const s = String(v ?? '').trim();
  if (!s) return NaN;
  if (/^\d{4}-\d{2}-\d{2}/.test(s)) {
    const t = Date.parse(s);
    return isNaN(t) ? NaN : t / 864e5;
  }
  if (/^\d{5}(?:\.\d+)?$/.test(s)) {
    const n = +s;
    if (n > 20000 && n < 80000) return n;
  }
  if (/^-?\d+(?:\.\d+)?$/.test(s)) return NaN;
  const t = Date.parse(s);
  return isNaN(t) ? NaN : t / 864e5;
}

export function numVal(v: unknown): number {
  if (v == null) return NaN;
  const s = String(v).trim().replace(/,/g, '').replace(/%$/, '');
  if (!s || s === '-') return NaN;
  const n = +s;
  return Number.isFinite(n) ? n : NaN;
}

export function normalizeTable(raw: unknown[][]): { cols: string[]; rows: Row[] } {
  if (!raw || !raw.length) return { cols: [], rows: [] };
  let head = raw[0].map(cellText);
  let body = raw.slice(1);
  if (!head.some((h) => h && isNaN(+h))) {
    head = head.map((_, i) => 'Column ' + (i + 1));
    body = raw;
  }
  body = body.map((r) =>
    r.map((v, j) => (/date|time/i.test(head[j] || '') ? excelSerialToISO(cellText(v)) : cellText(v)))
  );
  const keep = head.map((_, j) => body.some((r) => r[j] !== ''));
  head = head.filter((_, j) => keep[j]);
  body = body.map((r) => r.filter((_, j) => keep[j]));
  const seen: Record<string, number> = {};
  head = head.map((h, i) => {
    let n = (h || 'Column ' + (i + 1)).trim();
    if (seen[n]) {
      seen[n]++;
      n += ' (' + seen[n] + ')';
    } else seen[n] = 1;
    return n;
  });
  const rows = body
    .map((r) => Object.fromEntries(head.map((h, j) => [h, r[j] ?? ''])) as Row)
    .filter((r) => Object.values(r).some((v) => v !== ''));
  return { cols: head, rows };
}

let XLSX_NS: any = null;

async function loadSheetJS(): Promise<any> {
  if (XLSX_NS) return XLSX_NS;
  const urls = [
    'https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js',
    'https://unpkg.com/xlsx@0.18.5/dist/xlsx.full.min.js',
  ];
  let last: unknown = null;
  for (const u of urls) {
    try {
      await new Promise<void>((res, rej) => {
        const s = document.createElement('script');
        s.src = u;
        s.onload = () => res();
        s.onerror = () => rej(new Error('could not load ' + u));
        document.head.appendChild(s);
      });
      if (window.XLSX) {
        XLSX_NS = window.XLSX;
        break;
      }
    } catch (e) {
      last = e;
    }
  }
  if (!XLSX_NS) {
    const msg = last && (last as Error).message ? (last as Error).message : String(last || '');
    throw new Error('Excel reader unavailable' + (msg ? ' (' + msg + ')' : ''));
  }
  return XLSX_NS;
}

function readWorkbookSheet(ab: ArrayBuffer, name: string): { cols: string[]; rows: Row[] } {
  const xlsx = XLSX_NS;
  if (!xlsx) throw new Error('Workbook reader is not loaded.');
  const wb = xlsx.read(ab, { type: 'array', cellDates: true });
  const ws = wb.Sheets[name] || wb.Sheets[wb.SheetNames[0]];
  if (!ws) throw new Error('Sheet "' + name + '" was not found');
  const raw = xlsx.utils.sheet_to_json(ws, { header: 1, defval: '', raw: true, blankrows: false }) as unknown[][];
  return normalizeTable(raw);
}

function columnProfile(rows: Row[], c: string): ColumnProfile {
  const vals = rows.map((r) => String(r[c] ?? '')).filter((v) => v !== '');
  const us = new Set(vals);
  const num = vals.filter((v) => !isNaN(numVal(v))).length;
  const date = vals.length > 0 && vals.filter((v) => !isNaN(dateNum(v))).length >= 0.8 * vals.length;
  return {
    n: vals.length,
    uniques: us.size,
    ratio: vals.length ? us.size / vals.length : 0,
    numeric: vals.length > 0 && num >= 0.8 * vals.length,
    isDate: Boolean(date),
  };
}

export function profileData(cols: string[], rows: Row[]): Profile {
  const P: Record<string, ColumnProfile> = {};
  cols.forEach((c) => (P[c] = columnProfile(rows, c)));
  const idLike = cols.filter(
    (c) => P[c].ratio > 0.9 && P[c].n > 10 && !P[c].isDate && P[c].uniques > P[c].n * 0.5
  );
  const dates = cols.filter((c) => P[c].isDate);
  const texts = cols.filter(
    (c) => !P[c].numeric && !P[c].isDate && !idLike.includes(c) && P[c].uniques > 1
  );
  const cats = cols.filter(
    (c) =>
      P[c].uniques >= 2 &&
      P[c].uniques <= 50 &&
      P[c].ratio < 0.9 &&
      !idLike.includes(c) &&
      !dates.includes(c)
  );
  const outcome =
    cols.find((c) =>
      /^(completed|complete|done|status|result|label|class|target|outcome|approved|churn|won|success|fail|passed)/i.test(c)
    ) ||
    cats.find((c) => /complet|status|done|label|target|outcome|result/i.test(c)) ||
    cols[cols.length - 1] ||
    null;
  const descriptive = (c: string) =>
    c !== outcome &&
    /task|text|desc|title|body|content|review|comment|name|question|message|product|item/i.test(c) &&
    !/_id$|^id$|_code$/i.test(c);
  const textCol =
    (cols.includes('task_text') ? 'task_text' : texts.find(descriptive) ||
      texts.find((c) => c !== outcome && !/id/i.test(c)) ||
      texts.find((c) => c !== outcome) ||
      null);
  return {
    P,
    idLike,
    dates,
    texts,
    cats,
    outcome,
    textCol,
    primaryDate: dates[0] || null,
    target: outcome || cols[cols.length - 1] || '',
  };
}

interface DerivedFeature {
  name: string;
  f: (row: Row, index: number) => string | number;
}

function deriveFeatures(meta: Profile, exclude: string | undefined, rows: Row[]): DerivedFeature[] {
  const out: DerivedFeature[] = [];
  const ds = meta.dates.filter((d) => d !== exclude);
  if (ds.length >= 2) {
    const a = ds[0];
    const b = ds.find((d) => d !== a) as string;
    out.push({
      name: 'days_to_' + (b.replace(/_date$/, '') || 'event'),
      f: (r) => {
        const x = dateNum(r[a]);
        const y = dateNum(r[b]);
        return isNaN(x) || isNaN(y) ? '' : Math.round(y - x);
      },
    });
  }
  if (ds.length) {
    const d = ds[0];
    out.push({
      name: d + '_weekday',
      f: (r) => {
        const n = dateNum(r[d]);
        return isNaN(n) ? '' : new Date(n * 864e5).getUTCDay();
      },
    });
    out.push({
      name: d + '_month',
      f: (r) => {
        const n = dateNum(r[d]);
        return isNaN(n) ? '' : new Date(n * 864e5).getUTCMonth() + 1;
      },
    });
    out.push({ name: 'row_order', f: (_r, i) => i });
  }
  const tid = Object.keys(meta.P || {}).find(
    (c) => /_id$|^category|^type$/i.test(c) && meta.P[c].uniques <= 200 && meta.P[c].uniques >= 2
  );
  if (tid) {
    const cnt: Record<string, number> = {};
    rows.forEach((r) => {
      const k = String(r[tid] ?? '');
      cnt[k] = (cnt[k] || 0) + 1;
    });
    out.push({ name: 'frequency_' + tid, f: (r) => cnt[String(r[tid] ?? '')] || 1 });
  }
  if (meta.textCol) {
    const tc = meta.textCol;
    out.push({ name: 'text_length', f: (r) => String(r[tc] ?? '').length });
    out.push({
      name: 'word_count',
      f: (r) => String(r[tc] ?? '').split(/\s+/).filter(Boolean).length,
    });
  }
  return out;
}

export function prepareDataset(
  rawCols: string[],
  rawRows: Row[],
  opts: { target?: string; source?: string } = {}
): PreparedData {
  let cols = rawCols.slice();
  let rows = rawRows.map((r) => ({ ...r }));
  if (cols.length < 2 || rows.length < 10) {
    return {
      dataset: { cols, rows, target: cols[0] || '' },
      profile: {
        P: {},
        idLike: [],
        dates: [],
        texts: [],
        cats: [],
        outcome: null,
        textCol: null,
        primaryDate: null,
        target: cols[0] || '',
      },
      valid: false,
      message: `This sheet has ${rows.length} rows and ${cols.length} columns. At least 10 rows and 2 columns are needed for training.`,
    };
  }
  let meta = profileData(cols, rows);
  const comp = cols.find((c) => /complet.*date|finished.*date|closed.*date/i.test(c));
  let target = opts.target || meta.outcome || cols[cols.length - 1];
  if (comp && cols.filter((c) => meta.P[c].isDate).length > 1 && !cols.some((c) => /^completed$/i.test(c))) {
    rows = rows.map((r) => ({ ...r, completed: String(r[comp] ?? '').trim() ? 'yes' : 'no' }));
    cols = cols.concat('completed');
    meta.targetSource = comp;
    target = 'completed';
  }
  meta.target = target;
  deriveFeatures(meta, meta.targetSource, rows).forEach((x) => {
    cols = cols.concat(x.name);
    rows = rows.map((r, i) => ({ ...r, [x.name]: String(x.f(r, i)) }));
  });
  meta.rows = rows;
  meta.allCols = cols;
  return {
    dataset: { cols, rows, target },
    profile: meta,
    valid: true,
    message: `Loaded ${rows.length} rows and ${cols.length} columns${opts.source ? ' from ' + opts.source : ''}.`,
  };
}

export async function loadUploadedFile(
  file: File,
  onStatus?: (m: string) => void
): Promise<UploadedData> {
  const name = file.name.toLowerCase();
  const status = onStatus || (() => undefined);
  status('Reading ' + file.name + '…');
  const up: UploadedData = {
    name: file.name,
    names: [],
    sheets: {},
    sheet: '',
    data: {
      dataset: { cols: [], rows: [], target: '' },
      profile: profileData([], []),
      valid: false,
      message: '',
    },
  };
  if (/\.(xlsx|xlsm|xls|ods)$/.test(name)) {
    status('Loading Excel reader from CDN…');
    const xlsx = await loadSheetJS();
    status('Parsing workbook…');
    const ab = await file.arrayBuffer();
    const wb = xlsx.read(ab, { type: 'array', cellDates: true });
    up.buffer = ab;
    up.names = (wb.SheetNames as string[]).slice();
    if (!up.names.length) throw new Error('Workbook has no sheets');
    let best: { nm: string; score: number } | null = null;
    for (const nm of up.names) {
      const table = readWorkbookSheet(ab, nm);
      up.sheets[nm] = table;
      const score = table.rows.length * Math.min(20, table.cols.length);
      if (!best || score > best.score) best = { nm, score };
    }
    return selectUploadSheet(up, (best as { nm: string }).nm);
  }
  if (/\.(csv|txt|tsv|dat)$/.test(name)) {
    const text = (await file.text()).replace(/^\uFEFF/, '');
    const table = normalizeTable(parseCSV(text));
    up.names = [file.name];
    up.sheets[file.name] = table;
    return selectUploadSheet(up, file.name);
  }
  throw new Error('Unsupported type. Use CSV, TSV, XLSX, XLS or ODS.');
}

export function selectUploadSheet(up: UploadedData, name: string): UploadedData {
  if (!up.sheets[name] && up.buffer) up.sheets[name] = readWorkbookSheet(up.buffer, name);
  const table = up.sheets[name];
  if (!table) throw new Error('Sheet "' + name + '" was not found');
  if (!table.rows.length) throw new Error('Sheet is empty');
  return {
    ...up,
    sheet: name,
    data: prepareDataset(table.cols, table.rows, { source: up.name + ' · ' + name }),
  };
}

export function demoRows(): Row[] {
  const r = mulberry(7);
  const rows: Row[] = [];
  for (let i = 0; i < 300; i++) {
    const pr = ['Low', 'Medium', 'High'][r() * 3 | 0];
    const est = 1 + (r() * 12 | 0);
    const dd = 1 + (r() * 20 | 0);
    const tags = r() * 4 | 0;
    const sc = r() < 0.5 ? 'private' : 'team';
    const desc = r() < 0.6 ? 1 : 0;
    const s =
      (pr === 'High' ? 0.5 : pr === 'Low' ? -0.2 : 0.1) +
      (dd > 7 ? 0.5 : -0.4) -
      est * 0.08 +
      tags * 0.12 +
      desc * 0.3 +
      (sc === 'private' ? 0.2 : 0) +
      (r() - 0.5) * 1.2;
    const words =
      s > 0
        ? ['delivered', 'finished', 'shipped', 'smooth', 'completed', 'ahead', 'on time']
        : ['blocked', 'delayed', 'pending', 'stuck', 'overdue', 'needs review', 'at risk'];
    const txt =
      words[Math.floor(r() * words.length)] +
      ' ' + pr + ' priority ' + sc + ' task with ' + tags + ' tags and ' + est + ' hours' +
      (desc ? ' documented' : '');
    rows.push({
      priority: pr,
      scope: sc,
      est_hours: String(est),
      days_to_due: String(dd),
      tag_count: String(tags),
      has_desc: String(desc),
      task_text: txt,
      completed: s > 0 ? 'yes' : 'no',
    });
  }
  return rows;
}

/* ------------------------------------------------------------------ */
/* Feature encoding and classifiers                                    */
/* ------------------------------------------------------------------ */

export function featureColsFor(dataset: Dataset, profile: Profile, target: string): string[] {
  const blocked = new Set<string>([target, profile.targetSource, profile.targetBinnedFrom].filter(Boolean) as string[]);
  (profile.idLike || []).forEach((c) => blocked.add(c));
  dataset.cols.forEach((c) => {
    if (/date|time/i.test(c) && /(complet|finish|closed|resolved)/i.test(c)) blocked.add(c);
  });
  return dataset.cols.filter((c) => !blocked.has(c));
}

interface EncoderSpec {
  c?: string;
  synth?: string;
  num?: number;
  med?: number;
  m?: number;
  sd?: number;
  cats?: string[];
  width: number;
  f?: (row: Row) => string | number;
}

export interface Encoder {
  dim: number;
  specs: string[];
  transform: (row: Row) => number[];
}

export function makeEncoder(
  cols: string[],
  rows: Row[],
  tr: number[],
  opts: { target: string; textCol: string; meta: Profile }
): Encoder {
  const meta = opts.meta;
  const blocked = new Set<string>(
    [opts.target, meta.targetSource, meta.targetBinnedFrom].filter(Boolean) as string[]
  );
  (meta.idLike || []).forEach((c) => blocked.add(c));
  const specs: EncoderSpec[] = [];
  let budget = 180;
  for (const c of cols) {
    if (blocked.has(c)) continue;
    if (/date|time/i.test(c) && /(complet|finish|closed|resolved)/i.test(c)) continue;
    const vals = rows.map((r) => String(r[c] ?? ''));
    const ne = vals.filter((v) => v !== '');
    if (!ne.length) continue;
    const uniq = new Set(ne);
    const avg = ne.reduce((a, v) => a + v.length, 0) / ne.length;
    if (uniq.size < 2) continue;
    if (uniq.size > 60 || uniq.size / Math.max(1, ne.length) > 0.6 || avg > 120) continue;
    const tv = tr.map((i) => String(rows[i][c] ?? '')).filter((v) => v !== '');
    const nums = tv.map(numVal).filter((v) => !isNaN(v));
    if (nums.length >= 0.8 * Math.max(1, tv.length) && tv.length) {
      if (uniq.size > rows.length * 0.5 && rows.length > 20) continue;
      const s = nums.slice().sort((a, b) => a - b);
      const med = s.length ? s[s.length >> 1] : 0;
      const m = nums.reduce((a, b) => a + b, 0) / (nums.length || 1);
      const sd = Math.sqrt(nums.reduce((a, b) => a + (b - m) ** 2, 0) / (nums.length || 1)) || 1;
      specs.push({ c, num: 1, med, m, sd, width: 1 });
      budget--;
      continue;
    }
    if (budget <= 1) break;
    const cnt: Record<string, number> = {};
    tv.forEach((v) => (cnt[v] = (cnt[v] || 0) + 1));
    const cats = Object.keys(cnt)
      .sort((a, b) => cnt[b] - cnt[a])
      .slice(0, Math.min(20, budget));
    if (cats.length < 2) continue;
    specs.push({ c, num: 0, cats, width: cats.length });
    budget -= cats.length;
  }
  const tc = opts.textCol || meta.textCol;
  if (tc) {
    specs.push({ synth: 'len', width: 1, f: (r) => String(r[tc] ?? '').length });
    specs.push({
      synth: 'words',
      width: 1,
      f: (r) => String(r[tc] ?? '').split(/\s+/).filter(Boolean).length,
    });
    specs.push({ synth: 'digits', width: 1, f: (r) => (String(r[tc] ?? '').match(/\d/g) || []).length });
  }
  if (!specs.length) specs.push({ synth: 'constant', width: 1, f: () => 0 });
  return {
    dim: specs.reduce((a, s) => a + s.width, 0),
    specs: specs.map((s) => s.c || s.synth || 'text_stats'),
    transform(row: Row): number[] {
      const out: number[] = [];
      for (const s of specs) {
        if (s.synth && s.f) {
          out.push(Number(s.f(row)) || 0);
          continue;
        }
        if (s.num) {
          const v = numVal(row[s.c as string]);
          out.push(isNaN(v) ? (s.med || 0) : (v - (s.m || 0)) / (s.sd || 1));
        } else {
          (s.cats || []).forEach((cat) => out.push(String(row[s.c as string] ?? '') === cat ? 1 : 0));
        }
      }
      return out;
    },
  };
}

function softmax(z: number[]): number[] {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

function logistic(X: Matrix, y: number[], Xt: Matrix, K: number, store: Store): number[] {
  const D = X[0].length + 1;
  const W = Array.from({ length: K }, () => new Array<number>(D).fill(0));
  const sm = (x: number[]) => {
    const z = W.map((w) => w[0] + w.slice(1).reduce((a, b, j) => a + b * x[j], 0));
    return softmax(z);
  };
  for (let ep = 0; ep < 200; ep++) {
    const G = Array.from({ length: K }, () => new Array<number>(D).fill(0));
    X.forEach((x, i) => {
      const p = sm(x);
      for (let k = 0; k < K; k++) {
        const e = p[k] - (y[i] === k ? 1 : 0);
        G[k][0] += e;
        for (let j = 1; j < D; j++) G[k][j] += e * x[j - 1];
      }
    });
    for (let k = 0; k < K; k++)
      for (let j = 0; j < D; j++) W[k][j] -= 0.5 * (G[k][j] / X.length + (j ? 0.001 * W[k][j] : 0));
  }
  if (store) {
    store.proba = sm;
    store.predict = (x) => {
      const p = sm(x);
      return p.indexOf(Math.max(...p));
    };
  }
  return Xt.map((x) => {
    const p = sm(x);
    return p.indexOf(Math.max(...p));
  });
}

function knn(X: Matrix, y: number[], Xt: Matrix, K: number, store: Store): number[] {
  const one = (x: number[]) => {
    const d = X.map((r, i) => [r.reduce((a, v, j) => a + (v - x[j]) ** 2, 0), y[i]] as [number, number])
      .sort((a, b) => a[0] - b[0])
      .slice(0, 7);
    const v = new Array<number>(K).fill(0);
    d.forEach(([dd, l]) => (v[l] += 1 / (1 + dd)));
    return v.indexOf(Math.max(...v));
  };
  if (store) store.predict = one;
  return Xt.map(one);
}

function nb(X: Matrix, y: number[], Xt: Matrix, K: number, store: Store): number[] {
  const D = X[0].length;
  const st: Array<{ m: number[]; v: number[]; p: number }> = [];
  for (let k = 0; k < K; k++) {
    const r = X.filter((_, i) => y[i] === k);
    const n = r.length || 1;
    const m = Array.from({ length: D }, (_, j) => r.reduce((a, x) => a + x[j], 0) / n);
    const v = Array.from(
      { length: D },
      (_, j) => r.reduce((a, x) => a + (x[j] - m[j]) ** 2, 0) / n + 1e-3
    );
    st.push({ m, v, p: Math.log((r.length + 1) / (X.length + K)) });
  }
  const scores = (x: number[]) =>
    st.map((s) => {
      let l = s.p;
      for (let j = 0; j < D; j++) l += -0.5 * Math.log(s.v[j]) - (x[j] - s.m[j]) ** 2 / (2 * s.v[j]);
      return l;
    });
  const one = (x: number[]) => argmax(scores(x));
  if (store) {
    store.predict = one;
    store.proba = (x) => softmax(scores(x));
  }
  return Xt.map(one);
}

function gini(c: number[], n: number): number {
  return n ? 1 - c.reduce((a, v) => a + (v / n) ** 2, 0) : 0;
}

function buildTree(
  X: Matrix,
  y: number[],
  idx: number[],
  K: number,
  depth: number,
  maxD: number,
  mf: number,
  r: () => number
): Tree {
  const cnt = new Array<number>(K).fill(0);
  idx.forEach((i) => cnt[y[i]]++);
  const maj = cnt.indexOf(Math.max(...cnt));
  if (depth >= maxD || idx.length < 6 || cnt[maj] === idx.length) return { leaf: maj };
  const D = X[0].length;
  let fs = Array.from({ length: D }, (_, i) => i);
  if (mf < D) {
    fs.sort(() => r() - 0.5);
    fs = fs.slice(0, mf);
  }
  let best: { f: number; th: number } | null = null;
  let bg = gini(cnt, idx.length);
  for (const f of fs) {
    const vs = Array.from(new Set(idx.map((i) => X[i][f]))).sort((a, b) => a - b);
    if (vs.length < 2) continue;
    const step = Math.max(1, Math.floor(vs.length / 12));
    for (let s = 0; s < vs.length - 1; s += step) {
      const th = (vs[s] + vs[s + 1]) / 2;
      const L = new Array<number>(K).fill(0);
      const R = new Array<number>(K).fill(0);
      let nl = 0;
      idx.forEach((i) => {
        if (X[i][f] <= th) {
          L[y[i]]++;
          nl++;
        } else R[y[i]]++;
      });
      const nr = idx.length - nl;
      if (nl < 2 || nr < 2) continue;
      const g = (nl * gini(L, nl) + nr * gini(R, nr)) / idx.length;
      if (g < bg - 1e-9) {
        bg = g;
        best = { f, th };
      }
    }
  }
  if (!best) return { leaf: maj };
  const b = best as { f: number; th: number };
  const l = idx.filter((i) => X[i][b.f] <= b.th);
  const rr = idx.filter((i) => X[i][b.f] > b.th);
  return {
    f: b.f,
    th: b.th,
    l: buildTree(X, y, l, K, depth + 1, maxD, mf, r),
    r: buildTree(X, y, rr, K, depth + 1, maxD, mf, r),
  };
}

function predTree(t: Tree, x: number[]): number {
  return t.leaf !== undefined ? t.leaf : predTree((x[t.f as number] <= (t.th as number) ? t.l : t.r) as Tree, x);
}

function tree(X: Matrix, y: number[], Xt: Matrix, K: number, store: Store): number[] {
  const t = buildTree(X, y, X.map((_, i) => i), K, 0, 6, X[0].length, mulberry(13));
  if (store) store.predict = (x) => predTree(t, x);
  return Xt.map((x) => predTree(t, x));
}

function forest(X: Matrix, y: number[], Xt: Matrix, K: number, store: Store): number[] {
  const r = mulberry(11);
  const mf = Math.max(1, Math.round(Math.sqrt(X[0].length)));
  const ts: Tree[] = [];
  for (let t = 0; t < 25; t++) {
    const idx = X.map(() => r() * X.length | 0);
    ts.push(buildTree(X, y, idx, K, 0, 8, mf, r));
  }
  const one = (x: number[]) => {
    const v = new Array<number>(K).fill(0);
    ts.forEach((t) => v[predTree(t, x)]++);
    return v.indexOf(Math.max(...v));
  };
  if (store) store.predict = one;
  return Xt.map(one);
}

function thresholds(X: Matrix, f: number, m: number): number[] {
  const v = Array.from(new Set(X.map((r) => r[f]))).sort((a, b) => a - b);
  if (v.length < 2) return [];
  const o: number[] = [];
  const k = Math.min(m, v.length - 1);
  for (let i = 0; i < k; i++) {
    const j = Math.floor(((i + 1) * v.length) / (k + 1));
    if (j > 0 && j < v.length) o.push((v[j - 1] + v[j]) / 2);
  }
  return o;
}

function adaboost(X: Matrix, y: number[], Xt: Matrix, K: number, store: Store): number[] {
  const n = X.length;
  const D = X[0].length;
  const TH = Array.from({ length: D }, (_, f) => thresholds(X, f, 12));
  const S: Array<{ f: number; th: number; l: number; r: number; a: number }> = [];
  let w = new Array<number>(n).fill(1 / n);
  for (let t = 0; t < 50; t++) {
    let b: { f: number; th: number; l: number; r: number; err: number } | null = null;
    for (let f = 0; f < D; f++)
      for (const th of TH[f]) {
        const L = new Array<number>(K).fill(0);
        const R = new Array<number>(K).fill(0);
        for (let i = 0; i < n; i++) (X[i][f] <= th ? L : R)[y[i]] += w[i];
        const l = L.indexOf(Math.max(...L));
        const r = R.indexOf(Math.max(...R));
        const err = 1 - (L[l] + R[r]);
        if (!b || err < b.err) b = { f, th, l, r, err };
      }
    if (!b) break;
    const e = Math.min(Math.max(b.err, 1e-10), 1 - 1e-10);
    if (e >= 1 - 1 / K) break;
    const a = Math.log((1 - e) / e) + Math.log(K - 1);
    S.push({ ...b, a });
    let s = 0;
    for (let i = 0; i < n; i++) {
      if ((X[i][b.f] <= b.th ? b.l : b.r) !== y[i]) w[i] *= Math.exp(a);
      s += w[i];
    }
    w = w.map((v) => v / s);
  }
  const c = new Array<number>(K).fill(0);
  y.forEach((v) => c[v]++);
  const mj = c.indexOf(Math.max(...c));
  const one = (x: number[]) => {
    if (!S.length) return mj;
    const v = new Array<number>(K).fill(0);
    S.forEach((s) => (v[x[s.f] <= s.th ? s.l : s.r] += s.a));
    return v.indexOf(Math.max(...v));
  };
  if (store) store.predict = one;
  return Xt.map(one);
}

interface GBTree {
  v?: number;
  f?: number;
  th?: number;
  l?: GBTree;
  r?: GBTree;
}

function gboost(X: Matrix, y: number[], Xt: Matrix, K: number, store: Store): number[] {
  const n = X.length;
  const D = X[0].length;
  const lr = 0.2;
  const TH = Array.from({ length: D }, (_, f) => thresholds(X, f, 10));
  const F = X.map(() => new Array<number>(K).fill(0));
  const trees: GBTree[][] = [];
  const grow = (g: number[], h: number[], idx: number[], d: number): GBTree => {
    let G = 0;
    let H = 0;
    idx.forEach((i) => {
      G += g[i];
      H += h[i];
    });
    const leaf: GBTree = { v: (lr * G) / (H + 1e-3) };
    if (d >= 3 || idx.length < 8) return leaf;
    let b: { f: number; th: number; sc: number } | null = null;
    for (let f = 0; f < D; f++)
      for (const th of TH[f]) {
        let GL = 0;
        let HL = 0;
        let nl = 0;
        idx.forEach((i) => {
          if (X[i][f] <= th) {
            GL += g[i];
            HL += h[i];
            nl++;
          }
        });
        if (nl < 3 || idx.length - nl < 3) continue;
        const GR = G - GL;
        const HR = H - HL;
        const sc = (GL * GL) / (HL + 1) + (GR * GR) / (HR + 1) - (G * G) / (H + 1);
        if (!b || sc > b.sc) b = { f, th, sc };
      }
    if (!b || b.sc <= 1e-9) return leaf;
    const bb = b as { f: number; th: number; sc: number };
    return {
      f: bb.f,
      th: bb.th,
      l: grow(g, h, idx.filter((i) => X[i][bb.f] <= bb.th), d + 1),
      r: grow(g, h, idx.filter((i) => X[i][bb.f] > bb.th), d + 1),
    };
  };
  const pv = (t: GBTree, x: number[]): number => (t.v !== undefined ? t.v : pv((x[t.f as number] <= (t.th as number) ? t.l : t.r) as GBTree, x));
  const all = X.map((_, i) => i);
  for (let m = 0; m < 40; m++) {
    const P = F.map(softmax);
    const rd: GBTree[] = [];
    for (let k = 0; k < K; k++) {
      const g = new Array<number>(n);
      const h = new Array<number>(n);
      for (let i = 0; i < n; i++) {
        const p = P[i][k];
        g[i] = (y[i] === k ? 1 : 0) - p;
        h[i] = Math.max(p * (1 - p), 1e-6);
      }
      const t = grow(g, h, all, 0);
      rd.push(t);
      for (let i = 0; i < n; i++) F[i][k] += pv(t, X[i]);
    }
    trees.push(rd);
  }
  const one = (x: number[]) => {
    const s = new Array<number>(K).fill(0);
    trees.forEach((rd) => rd.forEach((t, k) => (s[k] += pv(t, x))));
    return s.indexOf(Math.max(...s));
  };
  if (store) store.predict = one;
  return Xt.map(one);
}

function catboost(X: Matrix, y: number[], Xt: Matrix, K: number, store: Store): number[] {
  const n = X.length;
  const D = X[0].length;
  const lr = 0.15;
  const lam = 3;
  const rnd = mulberry(9);
  const cand: Array<[number, number]> = [];
  const F = X.map(() => new Array<number>(K).fill(0));
  const trees: Array<{ sp: Array<[number, number]>; V: number[][] }> = [];
  for (let f = 0; f < D; f++) thresholds(X, f, 10).forEach((th) => cand.push([f, th]));
  for (let m = 0; m < 50; m++) {
    const g: number[][] = [];
    const h: number[][] = [];
    for (let i = 0; i < n; i++) {
      const p = softmax(F[i]);
      g.push(p.map((v, k) => (y[i] === k ? 1 : 0) - v));
      h.push(p.map((v) => Math.max(v * (1 - v), 1e-6)));
    }
    let leaf = new Array<number>(n).fill(0);
    const sp: Array<[number, number]> = [];
    for (let d = 0; d < 4; d++) {
      const L = 1 << (d + 1);
      const cs = cand.length > 120 ? cand.filter(() => rnd() < 120 / cand.length) : cand;
      let b: { f: number; th: number; sc: number } | null = null;
      for (const [f, th] of cs) {
        const G = Array.from({ length: L }, () => new Array<number>(K).fill(0));
        const H = Array.from({ length: L }, () => new Array<number>(K).fill(0));
        for (let i = 0; i < n; i++) {
          const l = leaf[i] * 2 + (X[i][f] > th ? 1 : 0);
          for (let k = 0; k < K; k++) {
            G[l][k] += g[i][k];
            H[l][k] += h[i][k];
          }
        }
        let sc = 0;
        for (let l = 0; l < L; l++) for (let k = 0; k < K; k++) sc += (G[l][k] * G[l][k]) / (H[l][k] + lam);
        if (!b || sc > b.sc) b = { f, th, sc };
      }
      if (!b) break;
      const bb = b as { f: number; th: number; sc: number };
      sp.push([bb.f, bb.th]);
      leaf = leaf.map((l, i) => l * 2 + (X[i][bb.f] > bb.th ? 1 : 0));
    }
    const L = 1 << sp.length;
    const G = Array.from({ length: L }, () => new Array<number>(K).fill(0));
    const H = Array.from({ length: L }, () => new Array<number>(K).fill(0));
    for (let i = 0; i < n; i++)
      for (let k = 0; k < K; k++) {
        G[leaf[i]][k] += g[i][k];
        H[leaf[i]][k] += h[i][k];
      }
    const V = G.map((gl, l) => gl.map((v, k) => (lr * v) / (H[l][k] + lam)));
    for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) F[i][k] += V[leaf[i]][k];
    trees.push({ sp, V });
  }
  const one = (x: number[]) => {
    const s = new Array<number>(K).fill(0);
    trees.forEach((t) => {
      let l = 0;
      t.sp.forEach(([f, th]) => (l = l * 2 + (x[f] > th ? 1 : 0)));
      for (let k = 0; k < K; k++) s[k] += t.V[l][k];
    });
    return s.indexOf(Math.max(...s));
  };
  if (store) store.predict = one;
  return Xt.map(one);
}

function svm(X: Matrix, y: number[], Xt: Matrix, K: number, store: Store): number[] {
  const D = X[0].length;
  const r = mulberry(5);
  const W = Array.from({ length: K }, () => new Array<number>(D + 1).fill(0));
  const lam = 0.01;
  for (let k = 0; k < K; k++) {
    const w = W[k];
    let t = 0;
    for (let ep = 0; ep < 15; ep++)
      for (let s = 0; s < X.length; s++) {
        const i = r() * X.length | 0;
        const eta = 1 / (lam * (++t + 100));
        const yy = y[i] === k ? 1 : -1;
        let m = w[0];
        for (let j = 0; j < D; j++) m += w[j + 1] * X[i][j];
        for (let j = 1; j <= D; j++) w[j] *= 1 - eta * lam;
        if (yy * m < 1) {
          w[0] += eta * yy * 0.1;
          for (let j = 0; j < D; j++) w[j + 1] += eta * yy * X[i][j];
        }
      }
  }
  const one = (x: number[]) => {
    let b = 0;
    let bs = -1e18;
    W.forEach((w, k) => {
      let m = w[0];
      for (let j = 0; j < D; j++) m += w[j + 1] * x[j];
      if (m > bs) {
        bs = m;
        b = k;
      }
    });
    return b;
  };
  if (store) store.predict = one;
  return Xt.map(one);
}

function metrics(yt: number[], p: number[], K: number): { acc: number; P: number; R: number; F: number } {
  const acc = yt.filter((v, i) => v === p[i]).length / yt.length;
  let P = 0;
  let R = 0;
  let F = 0;
  let c = 0;
  for (let k = 0; k < K; k++) {
    const tp = yt.filter((v, i) => v === k && p[i] === k).length;
    const fp = p.filter((v, i) => v === k && yt[i] !== k).length;
    const fn = yt.filter((v, i) => v === k && p[i] !== k).length;
    if (tp + fp + fn === 0) continue;
    const pr = tp / (tp + fp || 1);
    const re = tp / (tp + fn || 1);
    P += pr;
    R += re;
    F += pr + re ? (2 * pr * re) / (pr + re) : 0;
    c++;
  }
  return { acc, P: P / c, R: R / c, F: F / c };
}

/* ------------------------------------------------------------------ */
/* Tokenizers: real HF GPT token IDs with byte-level BPE fallback      */
/* ------------------------------------------------------------------ */

const HF_REPOS = {
  cl100k: { repo: 'DWDMaiMai/tiktoken_cl100k_base', label: 'GPT-3.5/4 cl100k_base' },
  o200k: { repo: 'wellflat/o200k_base_tokenizer', label: 'GPT-4o o200k_base' },
};

let HF_NS: any = null;
let HFTOK: any = null;
let HFTOK_KEY = '';

function hfIds(tk: any, text: string): number[] {
  const s = String(text ?? '');
  let o: any = null;
  if (tk && typeof tk === 'function') {
    try { o = tk(s); } catch { /* ignore */ }
  }
  if (!o && tk && typeof tk.encode === 'function') {
    try { o = tk.encode(s); } catch { /* ignore */ }
  }
  if (!o) return [];
  const r = o && o.input_ids !== undefined ? o.input_ids : o;
  const a = r && r.data !== undefined ? r.data : r;
  try {
    return Array.from(a as ArrayLike<number>, Number).filter((n) => Number.isFinite(n));
  } catch {
    return [];
  }
}

async function ensureHF(kind: 'cl100k' | 'o200k', onStatus?: (m: string) => void): Promise<any> {
  const cfg = HF_REPOS[kind] || HF_REPOS.cl100k;
  if (HFTOK && HFTOK_KEY === kind) return HFTOK;
  if (!HF_NS) {
    const urls = [
      'https://esm.sh/@huggingface/transformers@4.3.2',
      'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.web.js',
    ];
    let last: unknown = null;
    for (const u of urls) {
      if (onStatus) onStatus('Loading Transformers.js from CDN…');
      try {
        HF_NS = await import(/* @vite-ignore */ u);
        if (HF_NS && HF_NS.AutoTokenizer) break;
        HF_NS = null;
      } catch (e) {
        last = e;
      }
    }
    if (!HF_NS) {
      const msg = last && (last as Error).message ? (last as Error).message : String(last || '');
      throw new Error('Transformers.js CDN unavailable' + (msg ? ' (' + msg + ')' : ''));
    }
  }
  if (onStatus) onStatus('Downloading ' + cfg.label + ' from Hugging Face…');
  const tk = await HF_NS.AutoTokenizer.from_pretrained(cfg.repo, {
    progress_callback: (p: any) => {
      if (!onStatus) return;
      const st = typeof p === 'string' ? p : (p && p.status) || '';
      if (st)
        onStatus(
          'HF ' + st +
          (p && p.file ? ' · ' + p.file : '') +
          (p && p.progress != null ? ' · ' + Math.round(p.progress * 100) + '%' : '')
        );
    },
  });
  HFTOK = tk;
  HFTOK_KEY = kind;
  return tk;
}

function makeHFEncoder(tk: any, texts: string[], maxV: number): TextEncoderLike {
  const freq = new Map<number, number>();
  for (const t of texts) for (const id of hfIds(tk, t)) freq.set(id, (freq.get(id) || 0) + 1);
  const top = Array.from(freq.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, Math.max(2, maxV - 2))
    .map((x) => x[0]);
  const map = new Map<number, number>();
  const rev: Array<number | null> = [null, null];
  top.forEach((id, i) => {
    map.set(id, i + 2);
    rev[i + 2] = id;
  });
  const dec = (c: number) => {
    if (c === 0) return '<pad>';
    if (c === 1) return '<unk>';
    const r = rev[c];
    if (r == null) return '?';
    try { return String(tk.decode([r])).replace(/\n/g, '\\n'); } catch { return '?'; }
  };
  return {
    kind: 'hf',
    size: Math.max(4, top.length + 2),
    topCount: top.length,
    raw: (t) => hfIds(tk, t),
    encode: (t) => hfIds(tk, t).map((id) => (map.has(id) ? (map.get(id) as number) : 1)).slice(0, 64),
    show: (ids) => ids.map(dec).join(' '),
    decode: (ids) => {
      try {
        return tk.decode(ids.map((c) => rev[c]).filter((v): v is number => v != null), {
          skip_special_tokens: true,
        });
      } catch { return ''; }
    },
  };
}

const BPE_SPECIAL = ['<|pad|>', '<|endoftext|>', '<|im_start|>', '<|im_end|>'];
const BYTE_ID = (c: number) => c + 16;
const TOKEN_START = 320;

function bpeByteMaps(): { enc: Map<number, number>; dec: Map<number, number> } {
  const bs: number[] = [];
  const cs: number[] = [];
  for (let b = 33; b <= 126; b++) { bs.push(b); cs.push(b); }
  for (let b = 161; b <= 172; b++) { bs.push(b); cs.push(b); }
  for (let b = 174; b <= 255; b++) { bs.push(b); cs.push(b); }
  let n = 256;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(n++); }
  const enc = new Map<number, number>();
  const dec = new Map<number, number>();
  for (let i = 0; i < 256; i++) { enc.set(bs[i], cs[i]); dec.set(cs[i], bs[i]); }
  return { enc, dec };
}

const BMAP = bpeByteMaps();
const utf8 = (s: string): number[] => Array.from(new TextEncoder().encode(s));

function bpeSplit(s: string): string[] {
  try {
    return s.match(/'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu) || [];
  } catch {
    return s.match(/[A-Za-z]+|\d+|[^\sA-Za-z\d]+|\s+/g) || [];
  }
}

function trainBPE(texts: string[], maxVocab: number): TextEncoderLike & { tokenCount: number } {
  const counts = new Map<string, number>();
  for (const t of texts)
    for (const p of bpeSplit(String(t ?? ''))) {
      if (!p) continue;
      const k = utf8(p).map((b) => BMAP.enc.get(b) as number).join(',');
      counts.set(k, (counts.get(k) || 0) + 1);
    }
  const words: Array<{ s: number[]; c: number }> = [];
  for (const [k, c] of counts) words.push({ s: k.split(',').map(Number), c });
  let pairs = new Map<string, number>();
  const recount = () => {
    pairs = new Map<string, number>();
    for (const w of words)
      for (let i = 0; i < w.s.length - 1; i++) {
        const k = w.s[i] + ',' + w.s[i + 1];
        pairs.set(k, (pairs.get(k) || 0) + w.c);
      }
  };
  recount();
  const syms: number[][] = Array(TOKEN_START);
  const name: string[] = Array(TOKEN_START);
  const merges = new Map<string, { id: number; rank: number }>();
  BPE_SPECIAL.forEach((sp, i) => { syms[i] = []; name[i] = sp; });
  Array.from(BMAP.dec.keys()).sort((a, b) => a - b).forEach((code) => {
    const id = BYTE_ID(code);
    syms[id] = [code];
    name[id] = String.fromCharCode(code);
  });
  let next = TOKEN_START;
  while (next < maxVocab && pairs.size) {
    let best: { k: string; c: number } | null = null;
    for (const [k, c] of pairs) if (!best || c > best.c || (c === best.c && k < best.k)) best = { k, c };
    if (!best) break;
    const [a, b] = (best as { k: string }).k.split(',').map(Number);
    const id = next++;
    syms[id] = [
      ...syms[a < TOKEN_START ? BYTE_ID(a) : a],
      ...syms[b < TOKEN_START ? BYTE_ID(b) : b],
    ];
    name[id] = syms[id].map((c) => String.fromCharCode(c)).join('');
    merges.set((best as { k: string }).k, { id, rank: merges.size });
    for (const w of words) {
      const out: number[] = [];
      for (let i = 0; i < w.s.length; ) {
        if (i < w.s.length - 1 && w.s[i] + ',' + w.s[i + 1] === (best as { k: string }).k) {
          out.push(id);
          i += 2;
        } else out.push(w.s[i++]);
      }
      w.s = out;
    }
    recount();
  }
  const toId = (x: number) => (x < TOKEN_START ? BYTE_ID(x) : x);
  return {
    kind: 'builtin',
    size: next,
    topCount: next - TOKEN_START,
    tokenCount: next - TOKEN_START,
    raw: (text) => {
      const ids: number[] = [];
      for (const p of bpeSplit(String(text ?? ''))) for (const b of utf8(p)) ids.push(BMAP.enc.get(b) as number);
      return ids;
    },
    encode: (text, drop = 0) => {
      const ids: number[] = [];
      for (const p of bpeSplit(String(text ?? ''))) {
        const s = utf8(p).map((b) => BMAP.enc.get(b) as number);
        while (s.length > 1) {
          let bi = -1;
          let br = 1e9;
          for (let i = 0; i < s.length - 1; i++) {
            const m = merges.get(s[i] + ',' + s[i + 1]);
            if (m && m.rank < br && !(drop && Math.random() < drop)) {
              br = m.rank;
              bi = i;
            }
          }
          if (bi < 0) break;
          const m = merges.get(s[bi] + ',' + s[bi + 1]) as { id: number };
          s.splice(bi, 2, m.id);
        }
        for (const x of s) ids.push(toId(x));
      }
      return ids;
    },
    decode: (ids) => {
      const bytes: number[] = [];
      for (const id of ids) {
        if (id < BPE_SPECIAL.length || !syms[id]) continue;
        for (const c of syms[id]) {
          const b = BMAP.dec.get(c);
          if (b != null) bytes.push(b);
        }
      }
      try { return new TextDecoder().decode(Uint8Array.from(bytes)); } catch { return ''; }
    },
    show: (ids) => ids.map((id) => (name[id] !== undefined ? name[id] : '?')).join(' '),
  };
}

function builtinEncoder(texts: string[], maxV: number): TextEncoderLike {
  const tok = trainBPE(texts, maxV);
  return {
    kind: 'builtin',
    size: tok.size,
    topCount: tok.topCount,
    raw: (t) => tok.encode(t),
    encode: (t) => tok.encode(t).slice(0, 64),
    show: (ids) => tok.show(ids),
    decode: (ids) => tok.decode(ids),
  };
}

function makeLSTM(V: number, E: number, H: number, K: number, seed = 11): LSTMModel {
  const r = mulberry(seed);
  const rnd = (s: number) => (r() * 2 - 1) * s;
  const P: Record<string, Float32Array> = {
    Emb: new Float32Array(V * E),
    W: new Float32Array(4 * H * (E + H)),
    b: new Float32Array(4 * H),
    Wy: new Float32Array(K * H),
    by: new Float32Array(K),
  };
  for (let i = 0; i < P.Emb.length; i++) P.Emb[i] = rnd(0.08);
  for (let i = 0; i < P.W.length; i++) P.W[i] = rnd(0.05);
  for (let i = 0; i < P.Wy.length; i++) P.Wy[i] = rnd(0.08);
  const NK = ['Emb', 'W', 'b', 'Wy', 'by'];
  const G = NK.map((n) => new Float32Array(P[n].length));
  const M = NK.map((n) => new Float32Array(P[n].length));
  const R = NK.map((n) => new Float32Array(P[n].length));
  const D = E + H;
  const tan = Math.tanh;
  const sig = (x: number) => 1 / (1 + Math.exp(-x));
  let step = 0;

  function forward(
    seq: number[],
    cache?: boolean
  ): Float32Array | { prob: Float32Array; hs: Float32Array[]; cs: Float32Array[]; zs: Float32Array[] } {
    const hs = cache ? [new Float32Array(H)] : null;
    const cs = cache ? [new Float32Array(H)] : null;
    const zs = cache ? [] : null;
    let h = cache ? (hs as Float32Array[])[0] : new Float32Array(H);
    let c = cache ? (cs as Float32Array[])[0] : new Float32Array(H);
    for (let q = 0; q < seq.length; q++) {
      const z = new Float32Array(4 * H);
      const zin = new Float32Array(D);
      for (let j = 0; j < H; j++) zin[j] = h[j];
      const off = seq[q] * E;
      for (let j = 0; j < E; j++) zin[H + j] = P.Emb[off + j];
      for (let g = 0; g < 4; g++)
        for (let u = 0; u < H; u++) {
          let s = P.b[g * H + u];
          const base = g * H * D + u * D;
          for (let j = 0; j < D; j++) s += P.W[base + j] * zin[j];
          z[g * H + u] = s;
        }
      const hn = new Float32Array(H);
      const cn = new Float32Array(H);
      for (let u = 0; u < H; u++) {
        const i = sig(z[u]);
        const f = sig(z[H + u]);
        const o = sig(z[2 * H + u]);
        const gg = tan(z[3 * H + u]);
        cn[u] = f * c[u] + i * gg;
        hn[u] = o * tan(cn[u]);
      }
      if (cache) {
        (hs as Float32Array[]).push(hn);
        (cs as Float32Array[]).push(cn);
        (zs as Float32Array[]).push(z);
      }
      h = hn;
      c = cn;
    }
    const log = new Float32Array(K);
    let mx = -1e18;
    for (let k = 0; k < K; k++) {
      let s = P.by[k];
      const base = k * H;
      for (let u = 0; u < H; u++) s += P.Wy[base + u] * h[u];
      log[k] = s;
      if (s > mx) mx = s;
    }
    let sum = 0;
    for (let k = 0; k < K; k++) {
      log[k] = Math.exp(log[k] - mx);
      sum += log[k];
    }
    const prob = new Float32Array(K);
    for (let k = 0; k < K; k++) prob[k] = log[k] / sum;
    return cache ? { prob, hs: hs as Float32Array[], cs: cs as Float32Array[], zs: zs as Float32Array[] } : prob;
  }

  return {
    V, E, H, K, forward,
    train(seq, y, lr) {
      const cache = forward(seq, true) as { prob: Float32Array; hs: Float32Array[]; cs: Float32Array[]; zs: Float32Array[] };
      const { prob, hs, cs, zs } = cache;
      const loss = -Math.log(Math.max(prob[y], 1e-12));
      G.forEach((a) => a.fill(0));
      const dl = new Float32Array(K);
      for (let k = 0; k < K; k++) dl[k] = prob[k] - (y === k ? 1 : 0);
      const hL = hs[hs.length - 1];
      for (let k = 0; k < K; k++) {
        const d = dl[k];
        for (let u = 0; u < H; u++) G[3][k * H + u] += d * hL[u];
        G[4][k] += d;
      }
      let dh = new Float32Array(H);
      for (let u = 0; u < H; u++) {
        let s = 0;
        for (let k = 0; k < K; k++) s += P.Wy[k * H + u] * dl[k];
        dh[u] = s;
      }
      const dc = new Float32Array(H);
      for (let q = seq.length - 1; q >= 0; q--) {
        const z = zs[q];
        const hp = hs[q];
        const cp = cs[q];
        const cn = cs[q + 1];
        const dz = new Float32Array(4 * H);
        for (let u = 0; u < H; u++) {
          const i = sig(z[u]);
          const f = sig(z[H + u]);
          const o = sig(z[2 * H + u]);
          const gg = tan(z[3 * H + u]);
          const tc = tan(cn[u]);
          const dcu = dc[u] + dh[u] * o * (1 - tc * tc);
          const di = dcu * gg;
          const df = dcu * cp[u];
          const doo = dh[u] * tc;
          const dg = dcu * i;
          dz[u] = di * i * (1 - i);
          dz[H + u] = df * f * (1 - f);
          dz[2 * H + u] = doo * o * (1 - o);
          dz[3 * H + u] = dg * (1 - gg * gg);
          dc[u] = dcu * f;
        }
        const zin = new Float32Array(D);
        for (let j = 0; j < H; j++) zin[j] = hp[j];
        const off = seq[q] * E;
        for (let j = 0; j < E; j++) zin[H + j] = P.Emb[off + j];
        for (let g = 0; g < 4; g++)
          for (let u = 0; u < H; u++) {
            const gv = dz[g * H + u];
            const base = g * H * D + u * D;
            for (let j = 0; j < D; j++) G[1][base + j] += gv * zin[j];
            G[2][g * H + u] += gv;
          }
        const dzin = new Float32Array(D);
        for (let j = 0; j < D; j++) {
          let s = 0;
          for (let g = 0; g < 4; g++) for (let u = 0; u < H; u++) s += P.W[g * H * D + u * D + j] * dz[g * H + u];
          dzin[j] = s;
        }
        for (let j = 0; j < E; j++) G[0][off + j] += dzin[H + j];
        const nh = new Float32Array(H);
        for (let j = 0; j < H; j++) nh[j] = dzin[j];
        dh = nh;
      }
      let n2 = 0;
      for (const a of G) for (let i = 0; i < a.length; i++) n2 += a[i] * a[i];
      const sc = 5 / Math.max(5, Math.sqrt(n2));
      step++;
      const b1 = 0.9;
      const b2 = 0.999;
      const c1 = 1 - Math.pow(b1, step);
      const c2 = 1 - Math.pow(b2, step);
      for (let p = 0; p < 5; p++) {
        const A = P[NK[p]];
        const gr = G[p];
        const mm = M[p];
        const vv = R[p];
        for (let i = 0; i < A.length; i++) {
          const gi = gr[i] * sc;
          mm[i] = b1 * mm[i] + (1 - b1) * gi;
          vv[i] = b2 * vv[i] + (1 - b2) * gi * gi;
          A[i] -= (lr * (mm[i] / c1)) / (Math.sqrt(vv[i] / c2) + 1e-8);
        }
      }
      return loss;
    },
  };
}

/* ------------------------------------------------------------------ */
/* Unified training with best-model suggestions                        */
/* ------------------------------------------------------------------ */

function binTargetValues(rows: Row[], col: string): { name: string; values: string[] } | null {
  const vals = rows.map((r) => String(r[col] ?? '')).filter((v) => v !== '');
  if (!vals.length) return null;
  const nums = vals.map(numVal).filter((v) => !isNaN(v));
  if (nums.length >= 0.85 * vals.length && new Set(nums).size > 10) {
    const uniq = Array.from(new Set(nums)).sort((a, b) => a - b);
    const k = Math.min(8, Math.max(3, Math.round(Math.log2(uniq.length)) + 1));
    const cuts: number[] = [];
    for (let i = 1; i < k; i++) cuts.push(uniq[Math.floor((i * uniq.length) / k)]);
    return {
      name: col + ' (bins)',
      values: rows.map((r) => {
        const v = numVal(r[col]);
        if (isNaN(v)) return '';
        let b = 0;
        while (b < cuts.length && v >= cuts[b]) b++;
        const lo = b ? cuts[b - 1] : uniq[0];
        const hi = b < cuts.length ? cuts[b] : uniq[uniq.length - 1];
        return 'Q' + (b + 1) + ' (' + fmtNum(lo) + ' to ' + fmtNum(hi) + ')';
      }),
    };
  }
  const cnt: Record<string, number> = {};
  vals.forEach((v) => (cnt[v] = (cnt[v] || 0) + 1));
  const keys = Object.keys(cnt);
  if (keys.length > 10) {
    const top = keys.sort((a, b) => cnt[b] - cnt[a]).slice(0, 9);
    return { name: col + ' (bins)', values: rows.map((r) => (top.includes(String(r[col] ?? '')) ? String(r[col]) : 'Other')) };
  }
  return null;
}

interface Split {
  data: Row[];
  y: number[];
  tr: number[];
  te: number[];
  note: string;
  capped: boolean;
}

function splitData(rows: Row[], y: number[], mode: 'auto' | 'random' | 'chron', dateCol: string | null): Split {
  let data = rows.slice();
  let labels = y.slice();
  let capped = false;
  if (data.length > 1200) {
    capped = true;
    if (dateCol) {
      data = data
        .map((r, i) => ({ r, n: dateNum(r[dateCol]), i }))
        .sort((a, b) => (isNaN(a.n) ? -1 : a.n) - (isNaN(b.n) ? -1 : b.n))
        .slice(-1200)
        .map((x) => x.r);
      labels = data.map((r) => y[rows.indexOf(r)]);
    } else {
      const ord = shuffleIdx(data.length, 43);
      data = ord.slice(0, 1200).map((i) => data[i]);
      labels = data.map((r) => y[rows.indexOf(r)]);
    }
  }
  // Rebuild labels by value because capped rows may be reordered/copied.
  labels = data.map(() => 0);
  const originalY = new Map<Row, number>();
  rows.forEach((r, i) => originalY.set(r, y[i]));
  labels = data.map((r) => originalY.get(r) ?? y[rows.indexOf(r)] ?? 0);

  const randomSplit = (seed: number): Split => {
    const ord = shuffleIdx(data.length, seed);
    const d = ord.map((i) => data[i]);
    const yy = ord.map((i) => labels[i]);
    const nt = Math.max(2, Math.min(Math.round(d.length * 0.25), Math.floor(d.length / 2)));
    return {
      data: d,
      y: yy,
      tr: Array.from({ length: d.length - nt }, (_, i) => i + nt),
      te: Array.from({ length: nt }, (_, i) => i),
      note: 'deterministic random 75/25 split',
      capped,
    };
  };

  if (mode === 'chron' && dateCol) {
    const pairs = data
      .map((r, i) => ({ r, y: labels[i], n: dateNum(r[dateCol]) }))
      .sort((a, b) => (isNaN(a.n) ? -1 : a.n) - (isNaN(b.n) ? -1 : b.n));
    const nt = Math.max(2, Math.min(Math.round(pairs.length * 0.25), Math.floor(pairs.length / 2)));
    const sorted = pairs.slice(0, pairs.length - nt).concat(pairs.slice(pairs.length - nt));
    const s: Split = {
      data: sorted.map((x) => x.r),
      y: sorted.map((x) => x.y),
      tr: Array.from({ length: pairs.length - nt }, (_, i) => i),
      te: Array.from({ length: nt }, (_, i) => pairs.length - nt + i),
      note: 'chronological split on ' + dateCol + ' (earlier 75% → train, later 25% → test)',
      capped,
    };
    if (new Set(s.te.map((i) => s.y[i])).size < 2 || new Set(s.tr.map((i) => s.y[i])).size < 2) {
      const alt = randomSplit(42);
      alt.note = 'chronological test lacked both classes — used deterministic random split instead';
      return alt;
    }
    return s;
  }
  return randomSplit(42);
}

export async function trainAll(
  inputDataset: Dataset,
  inputProfile: Profile,
  options: TrainingOptions,
  onStatus: (m: string) => void,
  onProgress: (done: number, total: number, label: string) => void
): Promise<TrainingResult> {
  let dataset: Dataset = {
    cols: inputDataset.cols.slice(),
    rows: inputDataset.rows.map((r) => ({ ...r })),
    target: options.target,
  };
  let profile: Profile = { ...inputProfile };

  if (dataset.rows.length < 10 || dataset.cols.length < 2) throw new Error('Need at least 10 rows and 2 columns to train.');

  let target = options.target;
  let binned = false;
  if (!(profile.targetBinnedFrom && profile.target === target)) {
    const bins = binTargetValues(dataset.rows, target);
    if (bins) {
      target = bins.name;
      if (!dataset.cols.includes(target)) dataset.cols.push(target);
      dataset.rows = dataset.rows.map((r, i) => ({ ...r, [target]: bins.values[i] }));
      profile = { ...profile, target, targetBinnedFrom: options.target, rows: dataset.rows, allCols: dataset.cols };
      binned = true;
      dataset.target = target;
    }
  }

  const textCol = dataset.cols.includes(options.textCol)
    ? options.textCol
    : (profile.textCol && dataset.cols.includes(profile.textCol) ? profile.textCol : dataset.cols[0]);
  profile.target = target;
  dataset.target = target;

  const usableRows = dataset.rows.filter((r) => String(r[target] ?? '') !== '');
  if (usableRows.length < 10) throw new Error('Need at least 10 rows with a target value to train.');
  dataset = { ...dataset, rows: usableRows };
  profile = { ...profile, rows: dataset.rows, allCols: dataset.cols };

  const vals = dataset.rows.map((r) => String(r[target] ?? ''));
  const labels = Array.from(new Set(vals));
  if (labels.length < 2 || labels.length > 10)
    throw new Error('Target needs 2 to 10 distinct classes (found ' + labels.length + ').');

  const allY = dataset.rows.map((r) => labels.indexOf(String(r[target] ?? '')));
  const dateCol =
    options.split === 'chron' && profile.primaryDate && dataset.cols.includes(profile.primaryDate)
      ? profile.primaryDate
      : null;
  const split = splitData(dataset.rows, allY, options.split, dateCol);
  const data = split.data;
  const y = split.y;
  const tr = split.tr;
  const te = split.te;
  const K = labels.length;
  const ytr = tr.map((i) => y[i]);
  const yte = te.map((i) => y[i]);
  if (tr.length < 6 || te.length < 2) throw new Error('Need more rows for a train/test split.');
  if (new Set(yte).size < 2 || new Set(ytr).size < 2)
    throw new Error('Both train and test splits need at least two target classes.');

  const feats = featureColsFor(dataset, profile, target);
  const encoder = makeEncoder(feats, data, tr, { target, textCol, meta: profile });
  const Xtr = tr.map((i) => encoder.transform(data[i]));
  const Xte = te.map((i) => encoder.transform(data[i]));
  if (!Xtr[0] || !Xtr[0].length) throw new Error('No usable feature columns were found in this file.');

  const texts = data.map((r) => String(r[textCol] ?? ''));
  let enc: TextEncoderLike;
  let tokenizerNote = '';
  let tokenizerWarning = '';
  onStatus('Preparing tokenizer and features…');
  if (options.tokenizer === 'builtin') {
    enc = builtinEncoder(tr.map((i) => texts[i]), options.vocab);
    tokenizerNote = 'Built-in byte-level BPE (offline)';
  } else {
    try {
      onStatus('Loading real Hugging Face GPT tokenizer from CDN…');
      const tk = await ensureHF(options.tokenizer, onStatus);
      if (!hfIds(tk, 'tokenizer check').length) throw new Error('tokenizer produced no token IDs');
      enc = makeHFEncoder(tk, tr.map((i) => texts[i]), options.vocab);
      tokenizerNote =
        (options.tokenizer === 'o200k' ? 'GPT-4o o200k_base' : 'GPT-3.5/4 cl100k_base') +
        ' · ' + enc.topCount + ' frequent token IDs kept';
    } catch (e) {
      const msg = e && (e as Error).message ? (e as Error).message : String(e);
      tokenizerWarning = 'HF tokenizer unavailable (' + msg + ') — using built-in BPE.';
      onStatus(tokenizerWarning);
      enc = builtinEncoder(tr.map((i) => texts[i]), options.vocab);
      tokenizerNote = 'Built-in BPE fallback';
    }
  }

  const models: Array<[string, ModelFn]> = [
    ['Logistic Regression', logistic],
    ['k-Nearest Neighbors (k=7)', knn],
    ['Gaussian Naive Bayes', nb],
    ['Decision Tree', tree],
    ['Random Forest (25 trees)', forest],
    ['AdaBoost (SAMME, stumps)', adaboost],
    ['Gradient Boosting (trees)', gboost],
    ['CatBoost (oblivious trees)', catboost],
    ['Linear SVM (hinge loss)', svm],
  ];

  const results: Array<ModelMetric & { art: Artifact | null }> = [];
  for (let i = 0; i < models.length; i++) {
    onProgress(i, 10, 'Training ' + (i + 1) + '/10: ' + models[i][0]);
    onStatus('Training model ' + (i + 1) + ' of 10: ' + models[i][0]);
    await new Promise((r) => setTimeout(r, 18));
    const t0 = performance.now();
    const store: Store = {};
    try {
      const p = models[i][1](Xtr, ytr, Xte, K, store);
      const art: Artifact | null = store.predict
        ? {
            name: models[i][0],
            predict: (row) => (store.predict as (x: number[]) => number)(encoder.transform(row)),
            proba: store.proba ? (row) => (store.proba as (x: number[]) => number[])(encoder.transform(row)) : undefined,
          }
        : null;
      results.push({ n: models[i][0], ...metrics(yte, p, K), ms: Math.round(performance.now() - t0), art });
    } catch {
      results.push({ n: models[i][0], acc: 0, P: 0, R: 0, F: 0, ms: 0, err: true, art: null });
    }
  }

  onProgress(9, 10, 'Training 10/10: LSTM on ' + tokenizerNote);
  onStatus('Training 10/10: LSTM on ' + tokenizerNote);
  const net = makeLSTM(enc.size, options.bpeEmb, options.bpeHidden, K, 99);
  const seqs = texts.map((t) => {
    const a = enc.encode(t).slice(0, 48);
    return a.length ? a : [1];
  });
  const tIdx = tr.length > 600 ? shuffleIdx(tr.length, 77).slice(0, 600).map((i) => tr[i]) : tr.slice();
  const evaluate = (idx: number[]) => {
    let loss = 0;
    let cor = 0;
    const ps: number[] = [];
    for (const i of idx) {
      const p = net.forward(seqs[i]) as Float32Array;
      const pv = argmax(p);
      ps.push(pv);
      loss += -Math.log(Math.max(p[y[i]], 1e-12));
      if (pv === y[i]) cor++;
    }
    return { loss: loss / idx.length, acc: cor / idx.length, ps };
  };
  const history = { lt: [] as number[], le: [] as number[], at: [] as number[], ae: [] as number[] };
  const epochs = options.bpeEpochs > 0 ? options.bpeEpochs : options.epochs;
  for (let ep = 1; ep <= epochs; ep++) {
    for (const i of shuffleIdx(tIdx.length, 100 + ep).map((j) => tIdx[j])) net.train(seqs[i], y[i], 0.008);
    const a = evaluate(tIdx);
    const b = evaluate(te);
    history.lt.push(a.loss);
    history.le.push(b.loss);
    history.at.push(a.acc);
    history.ae.push(b.acc);
    onStatus('LSTM epoch ' + ep + '/' + epochs + ' · test accuracy ' + pct(b.acc));
    onProgress(9 + ep / epochs, 10, 'Training 10/10: LSTM epoch ' + ep + '/' + epochs);
    await new Promise((r) => setTimeout(r, 0));
  }

  const fin = evaluate(te);
  const mm = metrics(yte, fin.ps, K);
  const lname = 'LSTM · GPT token IDs';
  const seqFor = (row: Row) => {
    const a = enc.encode(String(row[textCol] ?? '')).slice(0, 48);
    return a.length ? a : [1];
  };
  const lstmArtifact: Artifact = {
    name: lname,
    predict: (row) => argmax(net.forward(seqFor(row)) as Float32Array),
    proba: (row) => Array.from(net.forward(seqFor(row)) as Float32Array),
  };
  results.push({ n: lname, ...mm, ms: 0, art: lstmArtifact });
  results.sort((a, b) => (a.err ? 1 : 0) - (b.err ? 1 : 0) || b.acc - a.acc || b.F - a.F);
  const best = results[0];
  if (!best || !best.art) throw new Error('No model produced a usable prediction.');
  const artifact = best.art;

  const idCol = (profile.idLike || []).find((c) => feats.includes(c) || dataset.cols.includes(c));
  const labelCols = (profile.cats || [])
    .filter((c) => c !== target && c !== profile.targetSource && dataset.cols.includes(c))
    .slice(0, 5);
  const pickIdx = te.slice(-8).reverse();
  const suggestions: Suggestion[] = pickIdx.map((idx) => {
    const row = data[idx];
    const p = artifact.predict(row);
    const conf = artifact.proba ? pct(artifact.proba(row)[p]) : '—';
    const fields = labelCols.map((c) => c + '=' + String(row[c] ?? '')).join(', ');
    return {
      id: idx,
      title: String(row[textCol] || (idCol ? row[idCol] : '') || 'row ' + (idx + 1)).slice(0, 90),
      label: labels[p] ?? String(p),
      conf,
      model: best.n,
      target,
      fields,
    };
  });

  const sample = data[te[0]];
  const sampleText = String(sample[textCol] ?? '');
  const rawSample = enc.raw(sampleText).slice(0, 40);
  const compactSample = enc.encode(sampleText).slice(0, 40);
  const baseline = yte.length
    ? Math.max(...labels.map((_, k) => yte.filter((v) => v === k).length)) / yte.length
    : 0;

  onProgress(10, 10, 'Done · best model: ' + best.n);
  onStatus('Done · best model: ' + best.n + ' (' + pct(best.acc) + ') · suggestions generated');

  return {
    dataset,
    profile,
    target,
    textCol,
    labels,
    features: feats,
    splitNote: split.note,
    capped: split.capped,
    trainSize: tr.length,
    testSize: te.length,
    tokenizerNote,
    tokenizerWarning,
    tokenizer: {
      example: sampleText,
      raw: rawSample,
      compact: compactSample,
      subwords: enc.show(compactSample),
      decoded: enc.decode(compactSample),
    },
    history,
    metrics: results.map(({ art: _art, ...rest }) => rest),
    best,
    artifact,
    suggestions,
    sample,
    baseline,
    binned,
  };
}

export function predictWith(result: TrainingResult, row: Row): { label: string; confidence: string; model: string } {
  const p = result.artifact.predict(row);
  const confidence = result.artifact.proba ? pct(result.artifact.proba(row)[p]) : '—';
  return { label: result.labels[p] ?? String(p), confidence, model: result.artifact.name };
}

/** Refresh derived columns after a user edits the text/date fields of a prediction row. */
export function syncPredictionRow(result: TrainingResult, raw: Row): Row {
  const row = { ...raw };
  const set = (name: string, value: string | number) => {
    if (value !== '' && result.dataset.cols.includes(name)) row[name] = String(value);
  };
  const text = String(row[result.textCol] ?? '');
  set('text_length', text.length);
  set('word_count', text.split(/\s+/).filter(Boolean).length);
  set('digits', (text.match(/\d/g) || []).length);

  const dates = (result.profile.dates || []).filter(
    (dateColumn) => dateColumn !== result.profile.targetSource && result.dataset.cols.includes(dateColumn)
  );
  dates.forEach((dateColumn) => {
    const n = dateNum(row[dateColumn]);
    if (isNaN(n)) return;
    set(dateColumn + '_weekday', new Date(n * 864e5).getUTCDay());
    set(dateColumn + '_month', new Date(n * 864e5).getUTCMonth() + 1);
  });
  if (dates.length >= 2) {
    const a = dates[0];
    const b = dates[1];
    const x = dateNum(row[a]);
    const y = dateNum(row[b]);
    if (!isNaN(x) && !isNaN(y)) {
      set('days_to_' + (b.replace(/_date$/, '') || 'event'), Math.round(y - x));
    }
  }
  return row;
}

/* ------------------------------------------------------------------ */
/* Schema-aware RAG / Agentic RAG / Graph RAG                           */
/* ------------------------------------------------------------------ */

const RAG_STOP = new Set(
  'the a an and or of to in for on with is are was were be by as at it this that from which who whom what when where how why do does did my me i we our you your their them these those has have had will would can could should about into over then also please show find give list tell tasks task data row rows complete completed done'.split(' ')
);

const ragTerms = (s: unknown): string[] =>
  String(s ?? '').toLowerCase().match(/[a-z0-9#@][a-z0-9#@_-]*/g) || [];

function ragExpand(q: string, profile: Profile, suggestions: Suggestion[]): string[] {
  const s = String(q || '').toLowerCase();
  const out: string[] = [];
  const add = (x: unknown) =>
    String(x ?? '').split(/[^a-z0-9_#@-]+/i).filter(Boolean).forEach((t) => out.push(t.toLowerCase()));
  if (/likely|predict|suggest|forecast|will|chance|risk|outcome|should|recommend/i.test(s)) {
    add(profile.target);
    add('prediction');
    add('suggestion');
    add('model');
    suggestions.slice(0, 3).forEach((x) => add(x.label));
  }
  if (/task|item|product|row|record|document|entry/i.test(s) && profile.textCol) add(profile.textCol);
  return Array.from(new Set(out));
}

/** Identifier-looking columns are useful for joins but noisy in RAG answers. */
const isIdColumn = (name: string): boolean => {
  const raw = String(name ?? '').trim();
  const lower = raw.toLowerCase();
  return (
    lower === 'id' ||
    lower === 'index' ||
    lower === 'row' ||
    lower === 'no' ||
    lower === 'number' ||
    /(^|[\s_-])ids?([\s_-]|$)/.test(lower) ||
    /[\s_-](?:ids?|index|row|order|no|number)$/.test(lower) ||
    /[a-z](?:Id|IDs|ID)$/.test(raw)
  );
};

export function ragDocs(dataset: Dataset | null, profile: Profile, suggestions: Suggestion[]): RAGDoc[] {
  if (!dataset || !dataset.rows.length) return [];
  const meta = profile;
  const searchableCols = dataset.cols.filter(
    (c) => !isIdColumn(c) && !(meta.idLike || []).includes(c)
  );
  const visibleCols = searchableCols.length ? searchableCols : dataset.cols;
  const preferredText =
    meta.textCol && visibleCols.includes(meta.textCol)
      ? meta.textCol
      : visibleCols.includes('task_text')
        ? 'task_text'
        : visibleCols.find((c) => /text|desc|title|body|content|review|comment|name/i.test(c)) ||
          visibleCols[0];
  const tc = preferredText || visibleCols[0] || dataset.cols[0];
  const labels = (meta.cats || [])
    .filter((c) => visibleCols.includes(c) && !isIdColumn(c) && c !== meta.targetSource)
    .slice(0, 8);
  const fromCol = dataset.cols.find((c) => /^(from|source|start)/i.test(c));
  const toCol = dataset.cols.find((c) => /^(to|destination|end)/i.test(c) && c !== fromCol);
  const docs: RAGDoc[] = dataset.rows.map((r, i) => {
    const title = String(
      r[tc] || visibleCols.map((c) => r[c]).find((value) => value !== undefined && value !== '') || 'Row ' + (i + 1)
    ).slice(0, 90);
    const tags = labels.map((c) => c + '=' + String(r[c] ?? '')).join(', ');
    return {
      id: i,
      title,
      text: visibleCols.map((c) => c + ': ' + String(r[c] ?? '')).join('. '),
      tags,
      priority: String(r.priority || r.status || (labels[0] ? r[labels[0]] : '') || ''),
      assignee: String(r.assignee || ''),
      edges:
        fromCol && toCol && String(r[fromCol] ?? '') && String(r[toCol] ?? '')
          ? [{ a: String(r[fromCol]), b: String(r[toCol]) }]
          : [],
    };
  });
  suggestions.forEach((s) => {
    const fields = String(s.fields || '')
      .split(',')
      .map((field) => field.trim())
      .filter((field) => field && !isIdColumn(field.split('=')[0]))
      .join(', ');
    docs.push({
      id: 'suggestion-' + s.id,
      title: 'Prediction · ' + s.title,
      text:
        'Model suggestion. model: ' + s.model + '. target: ' + s.target + '. predicted ' + s.target +
        ': ' + s.label + '. confidence: ' + s.conf + '. fields: ' + fields + '.',
      tags: fields,
      priority: s.label,
      assignee: s.model,
      edges: [],
      suggestion: true,
      label: s.label,
      conf: s.conf,
      model: s.model,
      target: s.target,
      sourceTitle: s.title,
    });
  });
  return docs;
}

interface IndexedDoc extends RAGDoc {
  tf: Map<string, number>;
  df: Map<string, number>;
  len: number;
  N: number;
}

function ragIndex(docs: RAGDoc[]): IndexedDoc[] {
  const df = new Map<string, number>();
  docs.forEach((d) => new Set(ragTerms(d.text)).forEach((t) => df.set(t, (df.get(t) || 0) + 1)));
  return docs.map((d) => {
    const terms = ragTerms(d.text);
    const tf = new Map<string, number>();
    terms.forEach((t) => tf.set(t, (tf.get(t) || 0) + 1));
    return { ...d, tf, df, len: Math.max(1, terms.length), N: docs.length };
  });
}

function ragRank(
  q: string,
  idx: IndexedDoc[],
  limit: number,
  profile: Profile,
  suggestions: Suggestion[]
): Array<{ doc: IndexedDoc; score: number }> {
  const terms = Array.from(
    new Set(
      ragTerms(q)
        .filter((t) => !RAG_STOP.has(t) && t.length > 1)
        .concat(ragExpand(q, profile, suggestions))
    )
  );
  if (!terms.length) return [];
  return idx
    .map((d) => {
      let s = 0;
      for (const t of terms) {
        const f = d.tf.get(t) || 0;
        if (!f) continue;
        const n = d.df.get(t) || 0;
        const idf = Math.log(1 + (d.N - n + 0.5) / (n + 0.5));
        s += (f * 2.2 * idf) / (f + 1.2);
      }
      return { doc: d, score: s };
    })
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit || 5);
}

function ragSentences(d: RAGDoc): string[] {
  return String(d.text).split(/\.\s+|\n+|·/).map((s) => s.trim()).filter((s) => s.length > 10);
}

function ragExtract(d: RAGDoc, q: string, profile: Profile, suggestions: Suggestion[]): string {
  const direct = new Set(ragTerms(q).filter((t) => !RAG_STOP.has(t)));
  const expanded = new Set(ragExpand(q, profile, suggestions));
  const ss = ragSentences(d);
  let best = ss[0] || d.text;
  let bs = -1;
  for (const s of ss) {
    const text = s.toLowerCase();
    let n = 0;
    for (const t of direct) if (text.includes(t)) n += 2;
    for (const t of expanded) if (!direct.has(t) && text.includes(t)) n += 1;
    if (n > bs) {
      bs = n;
      best = s;
    }
  }
  return best.slice(0, 300);
}

function ragSnippet(d: RAGDoc, q: string, profile: Profile, suggestions: Suggestion[]): string {
  if (d && d.suggestion)
    return String(d.sourceTitle || d.title) + ' → predicted ' + String(d.target || 'target') + ' = ' +
      String(d.label || '') + (d.conf && d.conf !== '—' ? ' (' + d.conf + ' confidence)' : '') +
      ' by ' + String(d.model || 'best model') + '.';
  return ragExtract(d, q, profile, suggestions);
}

function ragGraph(idx: IndexedDoc[]): { nodes: Map<string, number>; edges: Map<string, number> } {
  const nodes = new Map<string, number>();
  const edges = new Map<string, number>();
  const link = (a: string, b: string) => {
    if (!a || !b || a === b) return;
    const k = [a, b].sort().join('|');
    edges.set(k, (edges.get(k) || 0) + 1);
  };
  idx.forEach((d) => {
    const ents = Array.from(
      new Set([...ragTerms(d.tags), ...ragTerms(d.assignee), String(d.priority || '').toLowerCase()].filter(Boolean))
    ).slice(0, 10);
    d.ents = ents;
    ents.forEach((e) => {
      nodes.set(e, (nodes.get(e) || 0) + 1);
      link('doc:' + String(d.id), e);
    });
    for (let i = 0; i < ents.length; i++) for (let j = i + 1; j < ents.length; j++) link(ents[i], ents[j]);
    (d.edges || []).forEach((e) => {
      const edge = e as { a?: unknown; b?: unknown };
      link('flow:' + String(edge.a), 'flow:' + String(edge.b));
    });
  });
  return { nodes, edges };
}

function ragDecision(
  q: string,
  docs: RAGDoc[],
  idx: IndexedDoc[]
): { mode: string; why: string } {
  const flow = docs.some((d) => d.edges && d.edges.length);
  const { nodes } = ragGraph(idx);
  const shared = Array.from(nodes.entries()).filter(([, c]) => c >= 3).length;
  const multipart = /\band\b|\bor\b|\balso\b|\bthen\b|\?|,|;/i.test(q);
  if (flow || shared >= 4)
    return {
      mode: 'graph',
      why: (flow ? 'Flowchart edges found in the data' : 'Many shared tags/entities (' + shared + ')') +
        ' — graph traversal connects related documents.',
    };
  if (multipart)
    return { mode: 'agentic', why: 'The query has multiple parts, so it is split into sub-queries with a retrieve → reflect → answer loop.' };
  return { mode: 'rag', why: 'A focused single-topic query — direct retrieval with an extractive answer is the best fit.' };
}

export function runRAG(
  queryText: string,
  modeChoice: 'auto' | 'rag' | 'agentic' | 'graph',
  docs: RAGDoc[],
  profile: Profile,
  suggestions: Suggestion[]
): RAGResult {
  const query = String(queryText || '').trim();
  if (!query) return { mode: 'none', why: 'Type a question first.', plan: [], trail: '', ranked: [], answer: '' };
  const idx = ragIndex(docs);
  const dec = modeChoice === 'auto'
    ? ragDecision(query, docs, idx)
    : { mode: modeChoice, why: 'Manual override selected.' };
  const mode = dec.mode;
  let ranked: Array<{ doc: IndexedDoc; score: number }> = [];
  const plan: string[] = [];
  let trail = '';

  if (mode === 'graph') {
    const { nodes, edges } = ragGraph(idx);
    const hits = ragRank(query, idx, 8, profile, suggestions);
    let seeds = Array.from(
      new Set(ragTerms(query).concat(ragExpand(query, profile, suggestions)).filter((t) => nodes.has(t)))
    ).slice(0, 6);
    if (!seeds.length) seeds = hits.slice(0, 4).map((h) => 'doc:' + String(h.doc.id));
    const seen = new Set(seeds);
    const ids = new Set<string>();
    let frontier = seeds;
    for (let hop = 0; hop < 2; hop++) {
      const next: string[] = [];
      for (const a of frontier)
        for (const k of edges.keys()) {
          const [x, y] = k.split('|');
          if (x !== a && y !== a) continue;
          const b = x === a ? y : x;
          if (seen.has(b)) continue;
          seen.add(b);
          next.push(b);
          if (b.startsWith('doc:')) ids.add(b.slice(4));
        }
      frontier = next;
    }
    hits.forEach((h) => ids.add(String(h.doc.id)));
    ranked = idx
      .filter((d) => ids.has(String(d.id)))
      .map((doc) => ({
        doc,
        score: (hits.find((h) => String(h.doc.id) === String(doc.id)) || { score: 0 }).score,
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 8);
    trail = 'Graph seeds: ' + (seeds.join(', ') || 'none') + ' · ' + seen.size + ' nodes reached in 2 hops · ' + edges.size + ' weighted edges';
  } else if (mode === 'agentic') {
    const parts = query
      .split(/\s+(?:and|also|then|plus|while)\s+|\?|,|;/i)
      .map((s) => s.trim())
      .filter((s) => s.length > 3);
    const steps = parts.length > 1 ? parts : [query];
    const all = new Map<string | number, { doc: IndexedDoc; score: number }>();
    steps.forEach((p, i) => {
      let r = ragRank(p, idx, 2, profile, suggestions);
      if (!r.length) {
        const short = p.split(/\s+/).filter((w) => !RAG_STOP.has(w)).slice(0, 4).join(' ');
        r = ragRank(short, idx, 2, profile, suggestions);
        plan.push('Step ' + (i + 1) + ': no hits — reformulated to “' + short + '”');
      } else plan.push('Step ' + (i + 1) + ': ' + p + ' → ' + r.length + ' documents');
      r.forEach((h) => all.set(h.doc.id, h));
    });
    ranked = Array.from(all.values()).sort((a, b) => b.score - a.score).slice(0, 8);
  } else {
    ranked = ragRank(query, idx, 6, profile, suggestions);
  }

  const answer = ranked.length
    ? ranked.slice(0, 3).map((h) => ragSnippet(h.doc, query, profile, suggestions)).join(' ')
    : 'No relevant context found — try different keywords.';
  return {
    mode,
    why: dec.why,
    plan,
    trail,
    ranked: ranked.map((h) => ({
      title: String(h.doc.title).slice(0, 50),
      score: h.score,
      excerpt: ragSnippet(h.doc, query, profile, suggestions),
    })),
    answer,
  };
}

export function sparkline(values: number[], max: number): string {
  if (!values.length) return '';
  return values
    .map((v, i) => (12 + i * (296 / Math.max(1, values.length - 1))).toFixed(1) + ',' + (118 - (v / max) * 96).toFixed(1))
    .join(' ');
}
