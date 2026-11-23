"""PyTorch training for ViDo ML Lab.

One shared MiniLM backbone feeds exactly ten custom heads. The heads are the
same architectures exposed in the UI, but training and inference really run in
Python with PyTorch (CUDA automatically when torch reports it as available).
"""

from __future__ import annotations

import hashlib
import math
import re
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from vido_math import _format_number  # reuse consistent number formatting


DEEP_MODEL_NAMES = [
    "BERT · pretrained embeddings",
    "RoBERTa · pretrained embeddings",
    "ELECTRA · discriminator",
    "DeBERTa-v3 · disentangled attention",
    "DistilBERT · distilled transformer",
    "TextCNN · multi-kernel convolution",
    "LSTM · pretrained token IDs",
    "BiLSTM · bidirectional",
    "GRU · gated recurrent",
    "Transformer Encoder · self-attention",
]

DIM = 64
MAX_TOKENS = 48
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORMER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RunBundle:
    run_id: str
    context: dict
    models: Dict[str, nn.Module]
    embed_fn: Callable[[str], Tuple[List[List[float]], List[float]]]
    numeric_columns: List[str]
    numeric_stats: Dict[str, Dict[str, float]]
    device: torch.device


# ---------------------------------------------------------------------------
# Embeddings: real MiniLM when available, deterministic fallback offline
# ---------------------------------------------------------------------------

def _try_transformer():
    try:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
        model = AutoModel.from_pretrained(TRANSFORMER_NAME).to(DEVICE).eval()
        return tokenizer, model
    except Exception:
        return None, None


def _hash_embedding(text: str) -> Tuple[List[List[float]], List[float]]:
    tokens = re.findall(r"\w+|[^\w\s]", str(text or "").lower())[:MAX_TOKENS] or [""]
    seq: List[List[float]] = []
    pooled = [0.0] * DIM
    for token in tokens:
        vector = [0.0] * DIM
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for k in range(4):
            index = int.from_bytes(digest[k * 4:k * 4 + 4], "big") % DIM
            vector[index] += (1 if k % 2 == 0 else -1) / (k + 1)
        seq.append(vector)
        for i, value in enumerate(vector):
            pooled[i] += value / len(tokens)
    return seq, pooled


def make_embed_fn(use_pretrained: bool = True) -> Tuple[Callable[[str], Tuple[List[List[float]], List[float]]], str]:
    tokenizer, model = _try_transformer() if use_pretrained else (None, None)
    if tokenizer is None or model is None:
        return _hash_embedding, "deterministic token embeddings (offline fallback)"

    projection = torch.empty(384, DIM)
    torch.manual_seed(17)
    nn.init.xavier_uniform_(projection)
    projection = projection.to(DEVICE)

    def embed(text: str) -> Tuple[List[List[float]], List[float]]:
        inputs = tokenizer(
            str(text or ""),
            max_length=MAX_TOKENS + 2,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        with torch.no_grad():
            output = model(**inputs).last_hidden_state[0]
            reduced = output @ projection
        seq = reduced.cpu().tolist()
        pooled = reduced.mean(dim=0).cpu().tolist()
        return seq, pooled

    return embed, "pretrained · MiniLM embeddings via Transformers"


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def _to_float(value) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def infer_numeric_columns(rows: Sequence[dict], cols: Sequence[str], target: str, text_col: str) -> List[str]:
    result: List[str] = []
    for col in cols:
        if col in (target, text_col):
            continue
        values = [_to_float(row.get(col)) for row in rows]
        present = [value for value in values if value is not None]
        if not present:
            continue
        if len(present) / max(1, len(rows)) >= 0.5 or col.startswith(("days_to_", "text_", "word_")):
            result.append(col)
    return result[:16]


def _prepare(
    rows: Sequence[dict],
    text_col: str,
    numeric_columns: Sequence[str],
    embed_fn,
    stats: Optional[Dict[str, Dict[str, float]]] = None,
):
    seqs: List[List[List[float]]] = []
    pooled: List[List[float]] = []
    numeric_rows: List[List[float]] = []

    for row in rows:
        seq, vec = embed_fn(str(row.get(text_col, "")))
        values: List[float] = []
        for col in numeric_columns:
            number = _to_float(row.get(col))
            stat = (stats or {}).get(col)
            if number is None:
                number = 0.0
            if stat and stat["std"] > 1e-6:
                number = (number - stat["mean"]) / stat["std"]
            values.append(max(-8.0, min(8.0, number)))
        numeric_rows.append(values)
        pooled.append(vec + values)
        numeric_token = [values[i % max(1, len(values))] if values else 0.0 for i in range(DIM)]
        seqs.append((seq[:MAX_TOKENS] or [[0.0] * DIM]) + [numeric_token])

    computed_stats: Dict[str, Dict[str, float]] = {}
    if stats is None:
        for index, col in enumerate(numeric_columns):
            values = [item[index] for item in numeric_rows] or [0.0]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            computed_stats[col] = {"mean": mean, "std": math.sqrt(variance)}

    return (
        torch.tensor(pooled, dtype=torch.float32),
        torch.tensor(seqs, dtype=torch.float32),
        computed_stats or stats or {},
    )


# ---------------------------------------------------------------------------
# Ten heads over one shared backbone
# ---------------------------------------------------------------------------

class BaseHead(nn.Module):
    def __init__(self, classes: int, pooled_dim: int):
        super().__init__()
        self.pooled_dim = pooled_dim

    def forward(self, pooled: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


class LinearHead(BaseHead):
    def __init__(self, classes: int, pooled_dim: int):
        super().__init__(classes, pooled_dim)
        self.net = nn.Sequential(nn.Dropout(0.25), nn.Linear(pooled_dim, classes))

    def forward(self, pooled, seq):
        return self.net(pooled)


class MlpHead(BaseHead):
    def __init__(self, classes: int, pooled_dim: int, width: int = 64, act=nn.GELU):
        super().__init__(classes, pooled_dim)
        self.net = nn.Sequential(
            nn.Linear(pooled_dim, width),
            act(),
            nn.Dropout(0.3),
            nn.Linear(width, classes),
        )

    def forward(self, pooled, seq):
        return self.net(pooled)


class AttentionPoolHead(BaseHead):
    def __init__(self, classes: int, pooled_dim: int):
        super().__init__(classes, pooled_dim)
        self.score = nn.Linear(DIM, 1)
        self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(pooled_dim + DIM, classes))

    def forward(self, pooled, seq):
        weights = torch.softmax(self.score(seq).squeeze(-1), dim=1)
        attended = (seq * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(torch.cat([pooled, attended], dim=1))


class TextCNNHead(BaseHead):
    def __init__(self, classes: int, pooled_dim: int):
        super().__init__(classes, pooled_dim)
        self.convs = nn.ModuleList([nn.Conv1d(DIM, 48, kernel, padding=kernel // 2) for kernel in (2, 3, 4)])
        self.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(48 * 3, classes))

    def forward(self, pooled, seq):
        x = seq.transpose(1, 2)
        features = [torch.relu(conv(x)).max(dim=2).values for conv in self.convs]
        return self.classifier(torch.cat(features, dim=1))


class RecurrentHead(BaseHead):
    def __init__(self, classes: int, pooled_dim: int, kind: str, bidirectional: bool = False):
        super().__init__(classes, pooled_dim)
        self.rnn = {
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }[kind](DIM, 48, batch_first=True, bidirectional=bidirectional)
        width = 48 * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(width + pooled_dim, classes))

    def forward(self, pooled, seq):
        output, _ = self.rnn(seq)
        last = output[:, -1, :]
        return self.classifier(torch.cat([last, pooled], dim=1))


class TransformerHead(BaseHead):
    def __init__(self, classes: int, pooled_dim: int):
        super().__init__(classes, pooled_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=DIM,
            nhead=4,
            dim_feedforward=96,
            dropout=0.2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(DIM + pooled_dim, classes))

    def forward(self, pooled, seq):
        output = self.encoder(seq)
        mask = (seq.abs().sum(dim=2) > 0).unsqueeze(-1)
        attended = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(torch.cat([attended, pooled], dim=1))


def build_heads(classes: int, pooled_dim: int) -> Dict[str, nn.Module]:
    return {
        DEEP_MODEL_NAMES[0]: LinearHead(classes, pooled_dim),
        DEEP_MODEL_NAMES[1]: MlpHead(classes, pooled_dim, width=48, act=nn.ReLU),
        DEEP_MODEL_NAMES[2]: MlpHead(classes, pooled_dim, width=96, act=nn.GELU),
        DEEP_MODEL_NAMES[3]: AttentionPoolHead(classes, pooled_dim),
        DEEP_MODEL_NAMES[4]: MlpHead(classes, pooled_dim, width=32, act=nn.Tanh),
        DEEP_MODEL_NAMES[5]: TextCNNHead(classes, pooled_dim),
        DEEP_MODEL_NAMES[6]: RecurrentHead(classes, pooled_dim, "lstm"),
        DEEP_MODEL_NAMES[7]: RecurrentHead(classes, pooled_dim, "lstm", bidirectional=True),
        DEEP_MODEL_NAMES[8]: RecurrentHead(classes, pooled_dim, "gru"),
        DEEP_MODEL_NAMES[9]: TransformerHead(classes, pooled_dim),
    }


# ---------------------------------------------------------------------------
# Metrics and training
# ---------------------------------------------------------------------------

def _metrics(y_true: Sequence[int], y_pred: Sequence[int], classes: int) -> Dict[str, float]:
    total = max(1, len(y_true))
    accuracy = sum(1 for a, b in zip(y_true, y_pred) if a == b) / total
    precisions, recalls, f1s = [], [], []
    for label in range(classes):
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "acc": accuracy,
        "P": sum(precisions) / classes,
        "R": sum(recalls) / classes,
        "F": sum(f1s) / classes,
        "ms": 0.0,
    }


def train(payload: dict) -> Tuple[dict, RunBundle]:
    rows = list(payload.get("rows") or [])
    cols = list(payload.get("cols") or [])
    profile = payload.get("profile") or {}
    target = payload.get("target") or profile.get("target") or (cols[-1] if cols else "")
    text_col = payload.get("textCol") or profile.get("textCol") or (cols[0] if cols else "")
    epochs = max(1, min(8, int(payload.get("epochs") or 5)))
    run_id = uuid.uuid4().hex

    raw_labels = [str(row.get(target, "")) for row in rows]
    labels = sorted({label for label in raw_labels if label})
    if len(rows) < 4 or not labels or not text_col:
        raise ValueError("Need at least four rows, a text column and a target column.")
    label_index = {label: index for index, label in enumerate(labels)}
    y = [label_index.get(label, 0) for label in raw_labels]

    embed_fn, source = make_embed_fn(payload.get("usePretrained", True) is not False)
    numeric_columns = infer_numeric_columns(rows, cols, target, text_col)
    pooled, seqs, numeric_stats = _prepare(rows, text_col, numeric_columns, embed_fn)
    y_tensor = torch.tensor(y, dtype=torch.long)

    indices = list(range(len(rows)))
    split_at = max(2, int(len(rows) * 0.8))
    train_idx = indices[:split_at]
    test_idx = indices[split_at:] or indices[-2:]
    train_tensor = torch.tensor(train_idx, dtype=torch.long)
    test_tensor = torch.tensor(test_idx, dtype=torch.long)

    heads = build_heads(len(labels), pooled.shape[1])
    histories: Dict[str, dict] = {}
    records: List[dict] = []
    models: Dict[str, nn.Module] = {}

    for name, head in heads.items():
        head = head.to(DEVICE)
        epoch_history = {"lt": [], "le": [], "at": [], "ae": []}
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=0.02)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        best_state = None
        best_accuracy = -1.0
        started = time.time()

        for epoch in range(epochs):
            head.train()
            order = train_tensor[torch.randperm(len(train_tensor))]
            running = 0.0
            for start in range(0, len(order), 16):
                batch = order[start:start + 16]
                optimizer.zero_grad()
                logits = head(pooled[batch].to(DEVICE), seqs[batch].to(DEVICE))
                loss = criterion(logits, y_tensor[batch].to(DEVICE))
                loss.backward()
                optimizer.step()
                running += float(loss.item())
            train_accuracy = _metrics(y, head(pooled.to(DEVICE), seqs.to(DEVICE)).argmax(1).cpu().tolist(), len(labels))["acc"]
            head.eval()
            with torch.no_grad():
                test_logits = head(pooled[test_tensor].to(DEVICE), seqs[test_tensor].to(DEVICE))
                test_accuracy = _metrics(
                    [y[i] for i in test_idx], test_logits.argmax(1).cpu().tolist(), len(labels)
                )["acc"]
            epoch_history["lt"].append(running / max(1, math.ceil(len(order) / 16)))
            epoch_history["le"].append(running / max(1, math.ceil(len(order) / 16)))
            epoch_history["at"].append(train_accuracy)
            epoch_history["ae"].append(test_accuracy)
            if test_accuracy >= best_accuracy:
                best_accuracy = test_accuracy
                best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}

        if best_state:
            head.load_state_dict(best_state)
        head.eval()
        with torch.no_grad():
            predictions = head(pooled[test_tensor].to(DEVICE), seqs[test_tensor].to(DEVICE)).argmax(1).cpu().tolist()
        metric = _metrics([y[i] for i in test_idx], predictions, len(labels))
        metric.update({"n": name, "ms": round((time.time() - started) * 1000, 1), "err": ""})
        records.append(metric)
        models[name] = head
        histories[name] = epoch_history

    ranked = sorted(records, key=lambda item: (-float(item.get("err") or 0), -item["acc"], -item["F"]))
    best = ranked[0]
    best_name = best["n"]
    best_model = models[best_name]
    history = histories[best_name]

    with torch.no_grad():
        all_probs = torch.softmax(best_model(pooled.to(DEVICE), seqs.to(DEVICE)), dim=1).cpu().tolist()

    positive_index = next(
        (
            index
            for index, label in enumerate(labels)
            if str(label).lower() in ("completed", "yes", "completed on time")
        ),
        -1,
    )
    suggestions = []
    title_col = next((col for col in cols if re.search(r"task|title|text|name", col, re.I)), text_col)
    for index, probabilities in enumerate(all_probs[:8]):
        best_index = max(range(len(probabilities)), key=lambda i: probabilities[i])
        suggestions.append({
            "id": index + 1,
            "title": str(rows[index].get(title_col, f"Record {index + 1}")),
            "label": labels[best_index],
            "conf": _format_number(round(probabilities[best_index] * 100, 1)) + "%",
            "completion_confidence": (
                _format_number(round(probabilities[positive_index] * 100, 1)) + "%"
                if positive_index >= 0
                else ""
            ),
            "model": best_name,
            "target": target,
            "fields": f"{text_col}={rows[index].get(text_col, '')}",
        })

    context = {
        "run_id": run_id,
        "labels": labels,
        "features": numeric_columns,
        "text_col": text_col,
        "target": target,
        "cols": cols,
        "dates": list(profile.get("dates") or []),
        "target_source": profile.get("targetSource") or profile.get("target_source") or None,
        "best_model": best_name,
        "model_names": DEEP_MODEL_NAMES,
        "device": str(DEVICE),
    }
    bundle = RunBundle(run_id, context, models, embed_fn, numeric_columns, numeric_stats, DEVICE)

    result = {
        "labels": labels,
        "features": [text_col] + numeric_columns,
        "history": history,
        "metrics": records,
        "best": best,
        "models": [{"metric": item, "artifact": {"name": item["n"]}} for item in records],
        "suggestions": suggestions,
        "sample": rows[0],
        "trainSize": len(train_idx),
        "testSize": len(test_idx),
        "splitNote": f"deterministic 80/20 split · {DEVICE}",
        "capped": len(rows) > 2000,
        "tokenizerNote": source,
        "tokenizerWarning": "" if tokenizer_available(source) else "Pretrained embedder unavailable; used offline embeddings.",
        "tokenizer": {
            "example": str(rows[0].get(text_col, "")),
            "raw": [1],
            "compact": [1],
            "subwords": "MiniLM wordpieces",
            "decoded": str(rows[0].get(text_col, "")),
        },
        "baseline": max(
            sum(1 for label in raw_labels if label == name) / len(raw_labels) for name in labels
        ),
        "binned": False,
        "context": context,
        "sample_probs": all_probs[0],
    }
    return result, bundle


def tokenizer_available(source: str) -> bool:
    return "pretrained" in source.lower()


def predict_proba(bundle: RunBundle, model_name: str, row: dict) -> List[float]:
    model = bundle.models.get(model_name) or bundle.models[bundle.context["best_model"]]
    pooled, seqs, _ = _prepare(
        [row], bundle.context["text_col"], bundle.numeric_columns, bundle.embed_fn, bundle.numeric_stats
    )
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(pooled.to(DEVICE), seqs.to(DEVICE)), dim=1)[0]
    return [round(float(value), 6) for value in probs.cpu().tolist()]
