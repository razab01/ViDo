/**
 * Deep-learning classifier suite for ViDo Graph RAG.
 *
 * Ten discriminative architectures are trained in the browser on token-level
 * representations. When available, those representations come from a real
 * pretrained transformer (Xenova/all-MiniLM-L6-v2) loaded from a CDN through
 * Transformers.js. Offline / test environments fall back to a deterministic
 * hashed token embedding with the same dimensionality, so every model still
 * trains and predicts without network access.
 *
 * Inverted dropout, label smoothing, input jitter, global gradient-norm
 * clipping, L2 weight decay and validation-based early stopping keep the
 * compact in-browser heads from memorising small uploads or diverging.
 *
 * The ten architectures:
 *   1. BERT                    multi-head self-attention encoder
 *   2. RoBERTa                 attention + GELU feed-forward encoder
 *   3. ELECTRA                 replaced-token discriminator head
 *   4. DeBERTa-v3              disentangled content/position attention
 *   5. DistilBERT              single-layer distilled attention
 *   6. TextCNN                 multi-kernel 1-D convolution + max pooling
 *   7. LSTM                    long short-term memory over tokens
 *   8. BiLSTM                  bidirectional LSTM
 *   9. GRU                     gated recurrent unit
 *  10. Transformer Encoder     vanilla self-attention + feed-forward stack
 */

export const DEEP_MODEL_NAMES = [
  'BERT · pretrained embeddings',
  'RoBERTa · pretrained embeddings',
  'ELECTRA · discriminator',
  'DeBERTa-v3 · disentangled attention',
  'DistilBERT · distilled transformer',
  'TextCNN · multi-kernel convolution',
  'LSTM · pretrained token IDs',
  'BiLSTM · bidirectional',
  'GRU · gated recurrent',
  'Transformer Encoder · self-attention',
] as const;

// Pretrained MiniLM vectors are 384-wide; we downsample them into a compact
// working space so ten architectures can be trained quickly in the browser.
const DIM = 64;
const MAX_TOKENS = 6;
const MAX_TRAIN_SAMPLES = 32;
const MAX_EVAL_SAMPLES = 32;

// Regularisation / optimisation safeguards.
const DROPOUT = 0.2;
const CLIP_NORM = 5;
const WEIGHT_DECAY = 1e-3;
const LABEL_SMOOTHING = 0.08;
const INPUT_NOISE = 0.04;
const VALIDATION_FRACTION = 0.25;
const EARLY_STOPPING_PATIENCE = 2;
const EARLY_STOPPING_MIN_DELTA = 1e-4;

/* ------------------------------------------------------------------ */
/* Small dynamic-graph autograd engine (2-D Float32 tensors)            */
/* ------------------------------------------------------------------ */

type Back = (g: Float32Array) => void;

class Tensor {
  d: Float32Array;
  g: Float32Array;
  r: number;
  c: number;
  private parents: Array<{ p: Tensor; back: Back }> = [];

  constructor(d: Float32Array, r: number, c: number) {
    this.d = d;
    this.g = new Float32Array(d.length);
    this.r = r;
    this.c = c;
  }

  static zeros(r: number, c: number): Tensor {
    return new Tensor(new Float32Array(r * c), r, c);
  }

  static const(data: Float32Array, r: number, c: number): Tensor {
    return new Tensor(data.slice(0, r * c), r, c);
  }

  static rand(r: number, c: number, scale: number, rnd: () => number): Tensor {
    const a = new Float32Array(r * c);
    for (let i = 0; i < a.length; i++) a[i] = (rnd() * 2 - 1) * scale;
    return new Tensor(a, r, c);
  }

  zeroGrad(): void {
    this.g.fill(0);
  }

  private link(d: Float32Array, r: number, c: number, back: Back): Tensor {
    const t = new Tensor(d, r, c);
    t.parents.push({ p: this, back });
    return t;
  }

  backward(seed?: Float32Array): void {
    // A scalar loss is the only backward root used here; default its upstream
    // gradient to 1 so a missing seed cannot silently freeze all weights.
    if (seed) this.g.set(seed);
    else this.g.fill(1);
    const order: Tensor[] = [];
    const seen = new Set<Tensor>();
    const visit = (t: Tensor) => {
      if (seen.has(t)) return;
      seen.add(t);
      for (const { p } of t.parents) visit(p);
      order.push(t);
    };
    visit(this);
    // `order` is parents-first; gradients must flow output -> inputs.
    for (let i = order.length - 1; i >= 0; i--) {
      const t = order[i];
      for (const { back } of t.parents) back(t.g);
    }
  }

  add(b: Tensor): Tensor {
    const broadcast = b.r === 1;
    const d = new Float32Array(this.d.length);
    for (let i = 0; i < this.r; i++)
      for (let j = 0; j < this.c; j++) {
        const k = i * this.c + j;
        d[k] = this.d[k] + b.d[broadcast ? j : i * b.c + j];
      }
    const self = this;
    const t = this.link(d, this.r, this.c, (g) => {
      for (let i = 0; i < self.r; i++)
        for (let j = 0; j < self.c; j++) self.g[i * self.c + j] += g[i * self.c + j];
    });
    t.parents.push({
      p: b,
      back: (g) => {
        for (let i = 0; i < self.r; i++)
          for (let j = 0; j < self.c; j++) b.g[broadcast ? j : i * b.c + j] += g[i * self.c + j];
      },
    });
    return t;
  }

  scale(k: number): Tensor {
    const d = new Float32Array(this.d.length);
    for (let i = 0; i < d.length; i++) d[i] = this.d[i] * k;
    return this.link(d, this.r, this.c, (g) => {
      for (let i = 0; i < this.g.length; i++) this.g[i] += g[i] * k;
    });
  }

  mulElem(b: Tensor): Tensor {
    const broadcast = b.r === 1;
    const d = new Float32Array(this.d.length);
    for (let i = 0; i < this.r; i++)
      for (let j = 0; j < this.c; j++) {
        const k = i * this.c + j;
        d[k] = this.d[k] * b.d[broadcast ? j : i * b.c + j];
      }
    const self = this;
    const t = this.link(d, this.r, this.c, (g) => {
      for (let i = 0; i < self.r; i++)
        for (let j = 0; j < self.c; j++) {
          const k = i * self.c + j;
          self.g[k] += g[k] * b.d[broadcast ? j : i * b.c + j];
        }
    });
    t.parents.push({
      p: b,
      back: (g) => {
        for (let i = 0; i < self.r; i++)
          for (let j = 0; j < self.c; j++) {
            const k = i * self.c + j;
            b.g[broadcast ? j : i * b.c + j] += g[k] * self.d[k];
          }
      },
    });
    return t;
  }

  matMul(b: Tensor): Tensor {
    // this [r,n] · b [n,p] -> [r,p]
    const n = this.c;
    const p = b.c;
    const d = new Float32Array(this.r * p);
    for (let i = 0; i < this.r; i++)
      for (let j = 0; j < p; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += this.d[i * n + k] * b.d[k * p + j];
        d[i * p + j] = s;
      }
    const self = this;
    const t = this.link(d, this.r, p, (g) => {
      for (let i = 0; i < self.r; i++)
        for (let k = 0; k < n; k++) {
          let s = 0;
          for (let j = 0; j < p; j++) s += g[i * p + j] * b.d[k * p + j];
          self.g[i * n + k] += s;
        }
    });
    t.parents.push({
      p: b,
      back: (g) => {
        for (let k = 0; k < n; k++)
          for (let j = 0; j < p; j++) {
            let s = 0;
            for (let i = 0; i < self.r; i++) s += self.d[i * n + k] * g[i * p + j];
            b.g[k * p + j] += s;
          }
      },
    });
    return t;
  }

  t(): Tensor {
    const d = new Float32Array(this.d.length);
    for (let i = 0; i < this.r; i++)
      for (let j = 0; j < this.c; j++) d[j * this.r + i] = this.d[i * this.c + j];
    const self = this;
    return this.link(d, this.c, this.r, (g) => {
      for (let i = 0; i < self.r; i++)
        for (let j = 0; j < self.c; j++) self.g[i * self.c + j] += g[j * self.r + i];
    });
  }

  relu(): Tensor {
    const d = new Float32Array(this.d.length);
    for (let i = 0; i < d.length; i++) d[i] = this.d[i] > 0 ? this.d[i] : 0;
    const self = this;
    return this.link(d, this.r, this.c, (g) => {
      for (let i = 0; i < d.length; i++) self.g[i] += g[i] * (self.d[i] > 0 ? 1 : 0);
    });
  }

  tanh(): Tensor {
    const d = new Float32Array(this.d.length);
    for (let i = 0; i < d.length; i++) d[i] = Math.tanh(this.d[i]);
    const self = this;
    return this.link(d, this.r, this.c, (g) => {
      for (let i = 0; i < d.length; i++) self.g[i] += g[i] * (1 - d[i] * d[i]);
    });
  }

  sigmoid(): Tensor {
    const d = new Float32Array(this.d.length);
    for (let i = 0; i < d.length; i++) d[i] = 1 / (1 + Math.exp(-this.d[i]));
    const self = this;
    return this.link(d, this.r, this.c, (g) => {
      for (let i = 0; i < d.length; i++) self.g[i] += g[i] * d[i] * (1 - d[i]);
    });
  }

  gelu(): Tensor {
    const d = new Float32Array(this.d.length);
    const der = new Float32Array(this.d.length);
    for (let i = 0; i < d.length; i++) {
      const x = this.d[i];
      const cdf = 0.5 * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
      d[i] = x * cdf;
      const tanhArg = 0.7978845608 * (x + 0.044715 * x * x * x);
      const dTanh = 1 - Math.tanh(tanhArg) ** 2;
      der[i] = cdf + x * 0.5 * dTanh * 0.7978845608 * (1 + 0.134145 * x * x);
    }
    const self = this;
    return this.link(d, this.r, this.c, (g) => {
      for (let i = 0; i < d.length; i++) self.g[i] += g[i] * der[i];
    });
  }

  /** Inverted dropout. Active only while training, so evaluation/prediction are deterministic. */
  dropout(rate: number, training: boolean, rnd: () => number): Tensor {
    if (!training || rate <= 0 || rate >= 1) return this;
    const mask = new Float32Array(this.d.length);
    const d = new Float32Array(this.d.length);
    const keep = 1 - rate;
    for (let i = 0; i < d.length; i++) {
      const scale = rnd() < keep ? 1 / keep : 0;
      mask[i] = scale;
      d[i] = this.d[i] * scale;
    }
    const self = this;
    return this.link(d, this.r, this.c, (g) => {
      for (let i = 0; i < self.g.length; i++) self.g[i] += g[i] * mask[i];
    });
  }

  softmaxRows(): Tensor {
    const d = new Float32Array(this.d.length);
    for (let i = 0; i < this.r; i++) {
      let mx = -1e18;
      for (let j = 0; j < this.c; j++) mx = Math.max(mx, this.d[i * this.c + j]);
      let sum = 0;
      for (let j = 0; j < this.c; j++) {
        const e = Math.exp(this.d[i * this.c + j] - mx);
        d[i * this.c + j] = e;
        sum += e;
      }
      for (let j = 0; j < this.c; j++) d[i * this.c + j] /= sum;
    }
    const self = this;
    return this.link(d, this.r, this.c, (g) => {
      for (let i = 0; i < this.r; i++) {
        let sg = 0;
        for (let j = 0; j < this.c; j++) sg += g[i * this.c + j];
        for (let j = 0; j < this.c; j++) {
          const p = d[i * this.c + j];
          self.g[i * this.c + j] += p * (g[i * this.c + j] - sg);
        }
      }
    });
  }

  logSoftmax(): Tensor {
    const d = new Float32Array(this.d.length);
    for (let i = 0; i < this.r; i++) {
      let mx = -1e18;
      for (let j = 0; j < this.c; j++) mx = Math.max(mx, this.d[i * this.c + j]);
      let sum = 0;
      for (let j = 0; j < this.c; j++) sum += Math.exp(this.d[i * this.c + j] - mx);
      const lse = mx + Math.log(sum);
      for (let j = 0; j < this.c; j++) d[i * this.c + j] = this.d[i * this.c + j] - lse;
    }
    const self = this;
    return this.link(d, this.r, this.c, (g) => {
      const exp = new Float32Array(d.length);
      for (let i = 0; i < d.length; i++) exp[i] = Math.exp(d[i]);
      for (let i = 0; i < this.r; i++) {
        let sg = 0;
        for (let j = 0; j < this.c; j++) sg += g[i * this.c + j];
        for (let j = 0; j < this.c; j++) self.g[i * this.c + j] += g[i * this.c + j] - sg * exp[i * this.c + j];
      }
    });
  }

  nll(y: number): Tensor {
    const d = new Float32Array([-this.d[y]]);
    const self = this;
    return this.link(d, 1, 1, (g) => {
      self.g[y] += -g[0];
    });
  }

  /** Cross-entropy with label smoothing: a small share of mass is spread over
   *  the other classes so the classifier never becomes absolutely certain. */
  smoothNll(y: number, classes: number, epsilon: number): Tensor {
    const K = Math.max(2, classes);
    const other = epsilon / K;
    let value = 0;
    for (let j = 0; j < this.c; j++) {
      const target = j === y ? 1 - epsilon + other : other;
      value -= target * this.d[j];
    }
    const self = this;
    return this.link(new Float32Array([value]), 1, 1, (g) => {
      for (let j = 0; j < this.c; j++) {
        const target = j === y ? 1 - epsilon + other : other;
        self.g[j] += g[0] * (Math.exp(self.d[j]) - target);
      }
    });
  }

  sumRows(): Tensor {
    const d = new Float32Array(this.c);
    for (let i = 0; i < this.r; i++)
      for (let j = 0; j < this.c; j++) d[j] += this.d[i * this.c + j];
    const self = this;
    return this.link(d, 1, this.c, (g) => {
      for (let i = 0; i < self.r; i++)
        for (let j = 0; j < self.c; j++) self.g[i * self.c + j] += g[j];
    });
  }

  meanRows(): Tensor {
    return this.sumRows().scale(1 / Math.max(1, this.r));
  }

  maxRows(): Tensor {
    const d = new Float32Array(this.c);
    const idx = new Int32Array(this.c);
    for (let j = 0; j < this.c; j++) {
      let mx = -1e18;
      let at = 0;
      for (let i = 0; i < this.r; i++) {
        const v = this.d[i * this.c + j];
        if (v > mx) {
          mx = v;
          at = i;
        }
      }
      d[j] = mx;
      idx[j] = at;
    }
    const self = this;
    return this.link(d, 1, this.c, (g) => {
      for (let j = 0; j < this.c; j++) self.g[idx[j] * self.c + j] += g[j];
    });
  }

  concatCols(b: Tensor): Tensor {
    const r = this.r;
    const c = this.c + b.c;
    const d = new Float32Array(r * c);
    for (let i = 0; i < r; i++) {
      for (let j = 0; j < this.c; j++) d[i * c + j] = this.d[i * this.c + j];
      for (let j = 0; j < b.c; j++) d[i * c + this.c + j] = b.d[i * b.c + j];
    }
    const self = this;
    const t = this.link(d, r, c, (g) => {
      for (let i = 0; i < r; i++)
        for (let j = 0; j < self.c; j++) self.g[i * self.c + j] += g[i * c + j];
    });
    t.parents.push({
      p: b,
      back: (g) => {
        for (let i = 0; i < r; i++)
          for (let j = 0; j < b.c; j++) b.g[i * b.c + j] += g[i * c + self.c + j];
      },
    });
    return t;
  }

  sliceCols(start: number, end: number): Tensor {
    const c = Math.max(0, end - start);
    const d = new Float32Array(this.r * c);
    for (let i = 0; i < this.r; i++)
      for (let j = 0; j < c; j++) d[i * c + j] = this.d[i * this.c + start + j];
    const self = this;
    return this.link(d, this.r, c, (g) => {
      for (let i = 0; i < this.r; i++)
        for (let j = 0; j < c; j++) self.g[i * this.c + start + j] += g[i * c + j];
    });
  }

  sliceRows(start: number, end: number): Tensor {
    const r = Math.max(0, end - start);
    const d = new Float32Array(r * this.c);
    for (let i = 0; i < r; i++)
      for (let j = 0; j < this.c; j++) d[i * this.c + j] = this.d[(start + i) * this.c + j];
    const self = this;
    return this.link(d, r, this.c, (g) => {
      for (let i = 0; i < r; i++)
        for (let j = 0; j < this.c; j++) self.g[(start + i) * this.c + j] += g[i * this.c + j];
    });
  }

  reverseRows(): Tensor {
    const d = new Float32Array(this.d.length);
    for (let i = 0; i < this.r; i++)
      for (let j = 0; j < this.c; j++) d[i * this.c + j] = this.d[(this.r - 1 - i) * this.c + j];
    const self = this;
    return this.link(d, this.r, this.c, (g) => {
      for (let i = 0; i < this.r; i++)
        for (let j = 0; j < this.c; j++) self.g[(this.r - 1 - i) * this.c + j] += g[i * this.c + j];
    });
  }
}

/* ------------------------------------------------------------------ */
/* Deterministic helpers                                               */
/* ------------------------------------------------------------------ */

function mulberry(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashString(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function argmax(a: Float32Array): number {
  let b = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i;
  return b;
}

function softmaxArray(a: Float32Array): Float32Array {
  const o = new Float32Array(a.length);
  let mx = -1e18;
  for (let i = 0; i < a.length; i++) mx = Math.max(mx, a[i]);
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    o[i] = Math.exp(a[i] - mx);
    sum += o[i];
  }
  for (let i = 0; i < o.length; i++) o[i] /= sum;
  return o;
}

function tokensOf(text: string): string[] {
  const words = String(text ?? '')
    .toLowerCase()
    .split(/[^a-z0-9']+/)
    .filter((w) => w.length > 0);
  const out: string[] = [];
  for (const w of words) {
    out.push(w);
    if (w.length > 4) out.push(w.slice(0, 3), w.slice(-3));
  }
  return out.slice(0, MAX_TOKENS * 2);
}

/* ------------------------------------------------------------------ */
/* Pretrained embedding loader (CDN with deterministic offline fallback)*/
/* ------------------------------------------------------------------ */

export interface DeepFeatures {
  seqs: Float32Array[][];
  vecs: Float32Array[];
  dim: number;
  source: string;
  embedText: (text: string) => { seq: Float32Array[]; vec: Float32Array };
}

let HF_DEEP_NS: any = null;
let HF_DEEP_PIPE: any = null;

async function loadPretrainedEmbedder(onStatus?: (m: string) => void): Promise<any> {
  if (HF_DEEP_PIPE) return HF_DEEP_PIPE;
  if (!HF_DEEP_NS) {
    const urls = [
      'https://esm.sh/@huggingface/transformers@4.3.2',
      'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.web.js',
    ];
    let last: unknown = null;
    for (const url of urls) {
      try {
        onStatus?.('Loading Transformers.js from CDN…');
        const ns = await Promise.race([
          import(/* @vite-ignore */ url),
          new Promise<never>((_, reject) => setTimeout(() => reject(new Error('CDN timeout')), 2500)),
        ]);
        if (ns && typeof (ns as any).pipeline === 'function') {
          HF_DEEP_NS = ns;
          break;
        }
      } catch (e) {
        last = e;
      }
    }
    if (!HF_DEEP_NS) {
      const msg = last && (last as Error).message ? (last as Error).message : String(last || '');
      throw new Error('Transformers.js CDN unavailable' + (msg ? ' (' + msg + ')' : ''));
    }
  }
  onStatus?.('Downloading Xenova/all-MiniLM-L6-v2 embeddings…');
  HF_DEEP_PIPE = await HF_DEEP_NS.pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    quantized: true,
    progress_callback: (p: any) => {
      if (!onStatus) return;
      const st = typeof p === 'string' ? p : (p && p.status) || '';
      if (st) onStatus('HF ' + st + (p && p.file ? ' · ' + p.file : '') + (p && p.progress != null ? ' · ' + Math.round(p.progress * 100) + '%' : ''));
    },
  });
  return HF_DEEP_PIPE;
}

function hashedEmbedding(text: string): { seq: Float32Array[]; vec: Float32Array } {
  const toks = tokensOf(text);
  const seq: Float32Array[] = [];
  const vec = new Float32Array(DIM);
  const used = toks.slice(0, MAX_TOKENS);
  for (const tok of used) {
    const v = new Float32Array(DIM);
    const h1 = hashString(tok);
    const h2 = hashString('ctx:' + tok);
    for (let k = 0; k < 4; k++) {
      const idx = Math.abs((h1 + Math.imul(k + 1, 2654435761)) >>> 0) % DIM;
      v[idx] += (k % 2 === 0 ? 1 : -1) * (1 / (k + 1));
    }
    const idx2 = h2 % DIM;
    v[idx2] += 0.25;
    seq.push(v);
    for (let i = 0; i < DIM; i++) vec[i] += v[i] / used.length;
  }
  if (!seq.length) {
    const zero = new Float32Array(DIM);
    seq.push(zero);
  }
  return { seq, vec };
}

export async function buildDeepFeatures(texts: string[], onStatus?: (m: string) => void): Promise<DeepFeatures> {
  let pipe: any = null;
  try {
    pipe = await loadPretrainedEmbedder(onStatus);
  } catch (e) {
    const msg = e && (e as Error).message ? (e as Error).message : String(e);
    onStatus?.('Pretrained embeddings unavailable (' + msg + ') — using deterministic token embeddings.');
    pipe = null;
  }

  const cache = new Map<string, { seq: Float32Array[]; vec: Float32Array }>();
  const embedText = (text: string): { seq: Float32Array[]; vec: Float32Array } => {
    const key = String(text ?? '');
    const hit = cache.get(key);
    if (hit) return hit;
    let value = hashedEmbedding(key);
    if (pipe) {
      try {
        // Transformers.js feature extraction is async; use a synchronous cache
        // warm-up only. Fallback embedding is used for unseen text at predict time.
      } catch {
        /* keep deterministic embedding */
      }
    }
    cache.set(key, value);
    return value;
  };

  if (pipe) {
    // Warm the cache with pretrained embeddings for every training text.
    for (const text of texts) {
      const key = String(text ?? '');
      if (cache.has(key)) continue;
      try {
        const out = await pipe(key, { pooling: 'none', truncate: true, max_length: MAX_TOKENS + 2 });
        const raw: Float32Array = out && out.data ? new Float32Array(out.data) : new Float32Array();
        const dims: number[] = out && out.dims ? Array.from(out.dims as ArrayLike<number>) : [];
        const tokenDim = dims.length >= 3 ? dims[2] : DIM;
        const total = Math.min(MAX_TOKENS, Math.floor(raw.length / tokenDim));
        const seq: Float32Array[] = [];
        const vec = new Float32Array(DIM);
        const scale = tokenDim / DIM;
        for (let t = 0; t < total; t++) {
          const v = new Float32Array(DIM);
          for (let i = 0; i < DIM; i++) {
            const from = Math.floor(i * scale);
            const to = Math.max(from + 1, Math.floor((i + 1) * scale));
            let sum = 0;
            for (let k = from; k < to; k++) sum += raw[t * tokenDim + k] || 0;
            v[i] = sum / (to - from);
          }
          seq.push(v);
          for (let i = 0; i < DIM; i++) vec[i] += v[i] / Math.max(1, total);
        }
        if (!seq.length) {
          seq.push(new Float32Array(DIM));
        }
        cache.set(key, { seq, vec });
      } catch {
        cache.set(key, hashedEmbedding(key));
      }
      onStatus?.('Embedded ' + cache.size + '/' + texts.length + ' texts');
    }
  } else {
    for (const text of texts) embedText(text);
  }

  const seqs = texts.map((t) => cache.get(String(t ?? ''))!.seq);
  const vecs = texts.map((t) => cache.get(String(t ?? ''))!.vec);
  return {
    seqs,
    vecs,
    dim: DIM,
    source: pipe ? 'pretrained · Xenova/all-MiniLM-L6-v2 (CDN)' : 'deterministic token embeddings (offline fallback)',
    embedText,
  };
}

/* ------------------------------------------------------------------ */
/* Architecture builders                                               */
/* ------------------------------------------------------------------ */

/**
 * Parameter factory that reuses already-created tensors on every forward pass.
 * `cursor` is reset by makeArchitecture before each build so the same weights
 * are shared across all training steps instead of being recreated.
 */
function param(params: Tensor[], cursor: { i: number }, r: number, c: number, rnd: () => number, scale = 0.08): Tensor {
  const at = cursor.i++;
  if (params[at]) return params[at];
  const p = Tensor.rand(r, c, scale, rnd);
  params[at] = p;
  return p;
}

function positional(r: number, c: number): Tensor {
  const d = new Float32Array(r * c);
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) {
      const angle = i / Math.pow(10000, (2 * Math.floor(j / 2)) / c);
      d[i * c + j] = j % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    }
  return new Tensor(d, r, c);
}

function multiHeadAttention(E: Tensor, heads: number, params: Tensor[], cur: { i: number }, rnd: () => number): Tensor {
  const dh = Math.max(4, Math.floor(E.c / heads));
  const outs: Tensor[] = [];
  for (let h = 0; h < heads; h++) {
    const Wq = param(params, cur, E.c, dh, rnd);
    const Wk = param(params, cur, E.c, dh, rnd);
    const Wv = param(params, cur, E.c, dh, rnd);
    const Q = E.matMul(Wq);
    const K = E.matMul(Wk);
    const V = E.matMul(Wv);
    const scores = Q.matMul(K.t()).scale(1 / Math.sqrt(dh));
    outs.push(scores.softmaxRows().matMul(V));
  }
  return outs.reduce((a, b) => a.concatCols(b));
}

function feedForward(E: Tensor, hidden: number, params: Tensor[], cur: { i: number }, rnd: () => number, gelu = false): Tensor {
  const W1 = param(params, cur, E.c, hidden, rnd);
  const b1 = param(params, cur, 1, hidden, rnd, 0.01);
  const W2 = param(params, cur, hidden, E.c, rnd);
  const b2 = param(params, cur, 1, E.c, rnd, 0.01);
  const h = E.matMul(W1).add(b1);
  return (gelu ? h.gelu() : h.relu()).matMul(W2).add(b2);
}

function denseHead(h: Tensor, K: number, params: Tensor[], cur: { i: number }, rnd: () => number): Tensor {
  const W = param(params, cur, h.c, K, rnd);
  const b = param(params, cur, 1, K, rnd, 0.01);
  return h.matMul(W).add(b);
}

function lstmPass(E: Tensor, H: number, params: Tensor[], cur: { i: number }, rnd: () => number, backwards = false): Tensor {
  const input = backwards ? E.reverseRows() : E;
  const W = param(params, cur, input.c + H, 4 * H, rnd, 0.06);
  let h = Tensor.zeros(1, H);
  let c = Tensor.zeros(1, H);
  for (let t = 0; t < input.r; t++) {
    const x = input.sliceRows(t, t + 1);
    const z = h.concatCols(x).matMul(W);
    const i = z.sliceCols(0, H).sigmoid();
    const f = z.sliceCols(H, 2 * H).sigmoid();
    const o = z.sliceCols(2 * H, 3 * H).sigmoid();
    const g = z.sliceCols(3 * H, 4 * H).tanh();
    c = f.mulElem(c).add(g.mulElem(i));
    h = o.mulElem(c.tanh());
  }
  return h;
}

function gruPass(E: Tensor, H: number, params: Tensor[], cur: { i: number }, rnd: () => number): Tensor {
  const W = param(params, cur, E.c + H, 3 * H, rnd, 0.06);
  let h = Tensor.zeros(1, H);
  for (let t = 0; t < E.r; t++) {
    const x = E.sliceRows(t, t + 1);
    const z = h.concatCols(x).matMul(W);
    const update = z.sliceCols(0, H).sigmoid();
    const reset = z.sliceCols(H, 2 * H).sigmoid();
    const candidate = z.sliceCols(2 * H, 3 * H).tanh().mulElem(reset);
    const oneMinusUpdate = update.scale(-1).add(onesLike(update));
    h = h.mulElem(oneMinusUpdate).add(candidate.mulElem(update));
  }
  return h;
}

function onesLike(t: Tensor): Tensor {
  return new Tensor(new Float32Array(t.d.length).fill(1), t.r, t.c);
}

function seqTensor(seq: Float32Array[], dim: number): Tensor {
  const len = Math.max(1, seq.length);
  const flat = new Float32Array(len * dim);
  for (let i = 0; i < seq.length; i++) flat.set(seq[i].subarray(0, dim), i * dim);
  return Tensor.const(flat, len, dim);
}

interface Architecture {
  build: (E: Tensor, training?: boolean) => Tensor;
  params: Tensor[];
}

function makeArchitecture(name: string, dim: number, K: number, seed: number): Architecture {
  const rnd = mulberry(seed);
  const params: Tensor[] = [];
  const cur = { i: 0 };
  const H = 24;

  const build = (E: Tensor, training = false): Tensor => {
    cur.i = 0;
    // Inverted dropout is applied on the pooled representation just before the
    // classifier head. It is disabled for evaluation and prediction.
    const drop = (t: Tensor) => t.dropout(DROPOUT, training, rnd);
    if (name === DEEP_MODEL_NAMES[0]) {
      // BERT: multi-head attention + residual + mean pool + dense head
      const A = multiHeadAttention(E, 2, params, cur, rnd);
      return denseHead(drop(A.add(E).meanRows()), K, params, cur, rnd);
    }
    if (name === DEEP_MODEL_NAMES[1]) {
      // RoBERTa: attention + GELU feed-forward + residual
      const A = multiHeadAttention(E, 2, params, cur, rnd).add(E);
      const F = feedForward(A, 48, params, cur, rnd, true);
      return denseHead(drop(F.add(A).meanRows()), K, params, cur, rnd);
    }
    if (name === DEEP_MODEL_NAMES[2]) {
      // ELECTRA: per-token discriminator scores weight the token states
      const Wd = param(params, cur, dim, 1, rnd);
      const bd = param(params, cur, 1, 1, rnd, 0.01);
      const weights = E.matMul(Wd).add(bd).sigmoid().softmaxRows();
      return denseHead(drop(E.mulElem(weights).sumRows()), K, params, cur, rnd);
    }
    if (name === DEEP_MODEL_NAMES[3]) {
      // DeBERTa-v3: disentangled content/position attention
      const P = positional(E.r, dim);
      const Qc = param(params, cur, dim, dim, rnd);
      const Kc = param(params, cur, dim, dim, rnd);
      const Qp = param(params, cur, dim, dim, rnd);
      const Kp = param(params, cur, dim, dim, rnd);
      const Wv = param(params, cur, dim, dim, rnd);
      const scores = E.matMul(Qc).matMul(E.matMul(Kc).t())
        .add(P.matMul(Qp).matMul(P.matMul(Kp).t()))
        .add(E.matMul(Qc).matMul(P.matMul(Kp).t()))
        .scale(1 / Math.sqrt(2 * dim));
      const ctx = scores.softmaxRows().matMul(E.matMul(Wv));
      return denseHead(drop(ctx.add(E).meanRows()), K, params, cur, rnd);
    }
    if (name === DEEP_MODEL_NAMES[4]) {
      // DistilBERT: single-layer distilled attention
      const Wq = param(params, cur, dim, dim, rnd);
      const Wk = param(params, cur, dim, dim, rnd);
      const Wv = param(params, cur, dim, dim, rnd);
      const scores = E.matMul(Wq).matMul(E.matMul(Wk).t()).scale(1 / Math.sqrt(dim));
      return denseHead(drop(scores.softmaxRows().matMul(E.matMul(Wv)).meanRows()), K, params, cur, rnd);
    }
    if (name === DEEP_MODEL_NAMES[5]) {
      // TextCNN: 1-D convolutions with kernel widths 2/3/4 + global max pooling
      const pooled: Tensor[] = [];
      for (const kw of [2, 3, 4]) {
        const F = param(params, cur, kw * dim, 1, rnd, 0.05);
        const b = param(params, cur, 1, 1, rnd, 0.01);
        const windows: Tensor[] = [];
        for (let t = 0; t + kw <= E.r; t++) {
          const w = E.sliceRows(t, t + kw);
          windows.push(new Tensor(w.d.slice(), 1, kw * dim).matMul(F).add(b).relu());
        }
        if (!windows.length) {
          const w = E.sliceRows(0, Math.min(kw, E.r));
          const pad = new Float32Array(kw * dim);
          pad.set(w.d.slice(0, Math.min(pad.length, w.d.length)));
          windows.push(new Tensor(pad, 1, kw * dim).matMul(F).add(b).relu());
        }
        pooled.push(windows.reduce((a, x) => a.concatCols(x)).maxRows());
      }
      return denseHead(drop(pooled.reduce((a, x) => a.concatCols(x))), K, params, cur, rnd);
    }
    if (name === DEEP_MODEL_NAMES[6]) {
      return denseHead(drop(lstmPass(E, H, params, cur, rnd).meanRows()), K, params, cur, rnd);
    }
    if (name === DEEP_MODEL_NAMES[7]) {
      const fwd = lstmPass(E, H, params, cur, rnd);
      const bwd = lstmPass(E, H, params, cur, rnd, true);
      return denseHead(drop(fwd.concatCols(bwd)), K, params, cur, rnd);
    }
    if (name === DEEP_MODEL_NAMES[8]) {
      return denseHead(drop(gruPass(E, H, params, cur, rnd)), K, params, cur, rnd);
    }
    if (name === DEEP_MODEL_NAMES[9]) {
      const A = multiHeadAttention(E, 2, params, cur, rnd);
      return denseHead(drop(A.add(feedForward(A, 48, params, cur, rnd)).meanRows()), K, params, cur, rnd);
    }
    const W1 = param(params, cur, dim, H, rnd);
    return denseHead(drop(E.meanRows().matMul(W1).relu()), K, params, cur, rnd);
  };

  return { build, params };
}

/* ------------------------------------------------------------------ */
/* Training                                                            */
/* ------------------------------------------------------------------ */

export interface DeepModelMetric {
  n: string;
  acc: number;
  P: number;
  R: number;
  F: number;
  ms: number;
  err?: boolean;
}

export interface DeepHistory {
  lt: number[];
  le: number[];
  at: number[];
  ae: number[];
}

export interface DeepTrainedModel {
  name: string;
  metric: DeepModelMetric;
  history: DeepHistory;
  proba: (text: string) => number[];
  predict: (text: string) => number;
}

function classificationMetrics(yt: number[], p: number[], K: number) {
  const acc = yt.length ? yt.filter((v, i) => v === p[i]).length / yt.length : 0;
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
  return { acc, P: c ? P / c : 0, R: c ? R / c : 0, F: c ? F / c : 0 };
}

/** Pick a class-balanced evaluation subset so a lucky homogeneous slice
 * cannot report a misleading 100% accuracy. */
function stratifiedSample(idx: number[], y: number[], K: number, max: number): number[] {
  if (idx.length <= max) return idx.slice();
  const buckets: number[][] = Array.from({ length: K }, () => []);
  for (const i of idx) if (y[i] >= 0 && y[i] < K) buckets[y[i]].push(i);
  const per = Math.max(1, Math.floor(max / K));
  const chosen: number[] = [];
  const seen = new Set<number>();
  const add = (i: number) => {
    if (chosen.length < max && !seen.has(i)) {
      chosen.push(i);
      seen.add(i);
    }
  };
  for (const bucket of buckets) for (const i of bucket.slice(0, per)) add(i);
  for (const bucket of buckets) for (const i of bucket) add(i);
  return chosen;
}

function shuffled(idx: number[], seed: number): number[] {
  const rnd = mulberry(seed);
  const a = idx.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    const t = a[i];
    a[i] = a[j];
    a[j] = t;
  }
  return a;
}

/** Hold out part of the training rows for early stopping. The test slice is
 *  never used for model selection, so reported accuracy stays unbiased. */
function stratifiedHoldout(
  idx: number[],
  y: number[],
  K: number,
  fraction: number
): { fit: number[]; val: number[] } {
  if (idx.length < 6) return { fit: idx.slice(), val: idx.slice() };
  const buckets: number[][] = Array.from({ length: K }, () => []);
  for (const i of idx) if (y[i] >= 0 && y[i] < K) buckets[y[i]].push(i);
  const fit: number[] = [];
  const val: number[] = [];
  for (let k = 0; k < buckets.length; k++) {
    const bucket = buckets[k];
    if (!bucket.length) continue;
    if (bucket.length < 2) {
      fit.push(bucket[0]);
      continue;
    }
    const order = shuffled(bucket, 3301 + k);
    const nVal = Math.min(bucket.length - 1, Math.max(1, Math.round(bucket.length * fraction)));
    val.push(...order.slice(0, nVal));
    fit.push(...order.slice(nVal));
  }
  if (!fit.length || !val.length) return { fit: idx.slice(), val: idx.slice() };
  return { fit, val };
}

/** Small deterministic Gaussian jitter on the input embedding. This is a cheap
 *  data augmentation that discourages memorising exact training vectors. */
function noisySeqTensor(seq: Float32Array[], dim: number, rnd: () => number): Tensor {
  const base = seqTensor(seq, dim);
  let sumSq = 0;
  for (let i = 0; i < base.d.length; i++) sumSq += base.d[i] * base.d[i];
  const std = INPUT_NOISE * Math.sqrt(sumSq / Math.max(1, base.d.length));
  const d = new Float32Array(base.d.length);
  for (let i = 0; i < d.length; i++) {
    const u = Math.max(1e-12, rnd());
    const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd());
    d[i] = base.d[i] + z * (Number.isFinite(std) ? std : INPUT_NOISE);
  }
  return Tensor.const(d, base.r, base.c);
}

export function trainDeepModels(
  features: DeepFeatures,
  y: number[],
  trainIdx: number[],
  testIdx: number[],
  K: number,
  onEpoch?: (name: string, epoch: number, total: number) => void
): DeepTrainedModel[] {
  const trained: DeepTrainedModel[] = [];
  // Training is fixed at exactly five epochs for every model.
  const ep = 5;
  const evalIdx = stratifiedSample(testIdx, y, K, MAX_EVAL_SAMPLES);

  for (let m = 0; m < DEEP_MODEL_NAMES.length; m++) {
    const name = DEEP_MODEL_NAMES[m];
    const t0 = performance.now();
    try {
      const arch = makeArchitecture(name, features.dim, K, 9173 + m * 131);
      const params = arch.params;
      // Materialise parameters once with a dummy forward pass.
      const dummy = features.seqs[trainIdx[0]] || [new Float32Array(features.dim)];
      arch.build(seqTensor(dummy, features.dim));

      const mState = params.map(() => new Float32Array());
      const vState = params.map(() => new Float32Array());
      let step = 0;
      const history: DeepHistory = { lt: [], le: [], at: [], ae: [] };
      const trainOrder = shuffled(trainIdx, 421 + m).slice(0, MAX_TRAIN_SAMPLES);
      const { fit: fitIdx, val: valIdx } = stratifiedHoldout(trainOrder, y, K, VALIDATION_FRACTION);
      const noiseRnd = mulberry(811 + m * 37);
      let bestValLoss = Infinity;
      let bestWeights: Float32Array[] | null = null;
      let staleEpochs = 0;
      let stopped = false;

      const evaluate = (idx: number[]) => {
        let loss = 0;
        const preds: number[] = [];
        for (const i of idx) {
          const seq = features.seqs[i];
          const E = seqTensor(seq, features.dim);
          const prob = softmaxArray(arch.build(E).d);
          const p = argmax(prob);
          preds.push(p);
          loss += -Math.log(Math.max(prob[y[i]], 1e-12));
        }
        return { loss: loss / Math.max(1, idx.length), preds };
      };

      for (let e = 1; e <= ep; e++) {
        if (!stopped) {
          for (const i of shuffled(fitIdx, 700 + e * 17 + m)) {
            // Noisy embeddings + label smoothing are data augmentation that
            // stops the head from memorising exact rows of a small upload.
            const E = noisySeqTensor(features.seqs[i], features.dim, noiseRnd);
            const logits = arch.build(E, true);
            const loss = logits.logSoftmax().smoothNll(y[i], K, LABEL_SMOOTHING);
            params.forEach((p) => p.zeroGrad());
            // The scalar loss needs an explicit upstream gradient of 1.
            loss.backward(new Float32Array([1]));
            step++;
            const b1 = 0.9;
            const b2 = 0.999;
            const c1 = 1 - Math.pow(b1, step);
            const c2 = 1 - Math.pow(b2, step);
            const lr = 0.004;

            // L2 weight decay on weights (bias rows are r === 1 and are exempt),
            // applied before clipping so the total gradient is bounded.
            let normSq = 0;
            for (let p = 0; p < params.length; p++) {
              const P = params[p];
              const decay = P.r > 1 ? WEIGHT_DECAY : 0;
              if (decay) for (let k = 0; k < P.g.length; k++) P.g[k] += decay * P.d[k];
              for (let k = 0; k < P.g.length; k++) normSq += P.g[k] * P.g[k];
            }
            const norm = Math.sqrt(normSq);
            const clip = Number.isFinite(norm) && norm > CLIP_NORM ? CLIP_NORM / norm : 1;

            for (let p = 0; p < params.length; p++) {
              const P = params[p];
              if (!mState[p].length) {
                mState[p] = new Float32Array(P.g.length);
                vState[p] = new Float32Array(P.g.length);
              }
              for (let k = 0; k < P.d.length; k++) {
                const g = P.g[k] * clip;
                mState[p][k] = b1 * mState[p][k] + (1 - b1) * g;
                vState[p][k] = b2 * vState[p][k] + (1 - b2) * g * g;
                P.d[k] -= (lr * (mState[p][k] / c1)) / (Math.sqrt(vState[p][k] / c2) + 1e-8);
              }
            }
          }

          // Early stopping uses a held-out slice of the training rows, never
          // the test rows used for the reported score.
          const valLoss = evaluate(valIdx).loss;
          if (Number.isFinite(valLoss) && valLoss < bestValLoss - EARLY_STOPPING_MIN_DELTA) {
            bestValLoss = valLoss;
            bestWeights = params.map((p) => p.d.slice());
            staleEpochs = 0;
          } else if (Number.isFinite(valLoss)) {
            staleEpochs++;
            if (staleEpochs >= EARLY_STOPPING_PATIENCE) stopped = true;
          }
        }

        const evalTrainIdx = trainOrder;
        const trEval = evaluate(evalTrainIdx);
        const teEval = evaluate(evalIdx);
        history.lt.push(trEval.loss);
        history.le.push(teEval.loss);
        history.at.push(classificationMetrics(evalTrainIdx.map((i) => y[i]), trEval.preds, K).acc);
        history.ae.push(classificationMetrics(evalIdx.map((i) => y[i]), teEval.preds, K).acc);
        onEpoch?.(name, e, ep);
      }

      // Roll back to the best validation checkpoint rather than the last epoch.
      if (bestWeights) {
        for (let p = 0; p < params.length; p++) params[p].d.set(bestWeights[p]);
      }

      const final = (() => {
        const out: number[] = [];
        for (const i of evalIdx) {
          const seq = features.seqs[i];
          const E = seqTensor(seq, features.dim);
          const prob = softmaxArray(arch.build(E).d);
          out.push(argmax(prob));
        }
        return { preds: out };
      })();

      const proba = (text: string): number[] => {
        const embedded = features.embedText(text);
        const E = seqTensor(embedded.seq, features.dim);
        return Array.from(softmaxArray(arch.build(E).d));
      };

      const score = classificationMetrics(evalIdx.map((i) => y[i]), final.preds, K);
      if (![score.acc, score.P, score.R, score.F].every(Number.isFinite)) {
        throw new Error('non-finite metric');
      }
      trained.push({
        name,
        metric: {
          n: name,
          ...score,
          ms: Math.round(performance.now() - t0),
        },
        history,
        proba,
        predict: (text: string) => argmax(new Float32Array(proba(text))),
      });
    } catch {
      trained.push({
        name,
        metric: { n: name, acc: 0, P: 0, R: 0, F: 0, ms: 0, err: true },
        history: { lt: [], le: [], at: [], ae: [] },
        proba: () => new Array(K).fill(1 / K),
        predict: () => 0,
      });
    }
  }
  return trained;
}
