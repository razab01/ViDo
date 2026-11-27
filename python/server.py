"""FastAPI bridge between the React app and the Python ML/Graph-RAG stack."""

from __future__ import annotations

import base64
import binascii
import os
import sys
import tempfile
from typing import Dict, Optional, Tuple

# Keep a local math.py from shadowing Python's standard library module.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.append(HERE)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import ml
from document_text import DocumentError, extract_text
from graph_rag import answer_with_booking
from ml import RunBundle


app = FastAPI(title="ViDo Python ML", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS: Dict[str, RunBundle] = {}


class TrainBody(BaseModel):
    rows: list
    cols: list
    profile: dict = {}
    target: str = ""
    textCol: str = ""
    epochs: int = 5
    usePretrained: bool = True


class PredictBody(BaseModel):
    context: dict
    model: str = ""
    row: dict


class AnswerBody(BaseModel):
    question: str
    rows: list = []
    cols: list = []
    profile: dict = {}
    suggestions: list = []
    trained: Optional[dict] = None
    documents: list = []
    booking: Optional[dict] = None


class DocumentBody(BaseModel):
    name: str
    content: str


class TranscribeBody(BaseModel):
    name: str = "voice.webm"
    content: str
    language: Optional[str] = None


MAX_AUDIO_BYTES = 12 * 1024 * 1024
_ASR_CACHE: Dict[str, object] = {}


def _asr_suffix(name: str) -> str:
    lowered = name.lower()
    for suffix in (".webm", ".ogg", ".m4a", ".mp4", ".wav", ".mp3"):
        if lowered.endswith(suffix):
            return suffix
    return ".webm"


def _asr_model() -> Tuple[str, object]:
    """Load a tiny CPU speech model lazily so the app starts without it."""
    model_name = os.environ.get("VIDO_ASR_MODEL", "tiny").strip() or "tiny"
    cached = _ASR_CACHE.get(model_name)
    if cached is not None:
        return model_name, cached
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Offline voice is not installed on this device yet.",
        ) from exc
    try:
        # CPU int8 keeps transcription local and avoids any browser GPU work.
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="The offline speech model could not be started.",
        ) from exc
    _ASR_CACHE[model_name] = model
    return model_name, model


@app.get("/api/python/health")
def health():
    return {
        "ok": True,
        "torch": ml.torch.__version__,
        "device": str(ml.DEVICE),
        "cuda": ml.torch.cuda.is_available(),
    }


@app.post("/api/python/train")
def train(body: TrainBody):
    try:
        result, bundle = ml.train(body.model_dump(by_alias=True))
        RUNS[bundle.run_id] = bundle
        return result
    except Exception as exc:  # surface a readable message to the UI
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/python/predict")
def predict(body: PredictBody):
    run_id = str(body.context.get("run_id") or "")
    bundle = RUNS.get(run_id)
    if not bundle:
        raise HTTPException(status_code=404, detail="Training run expired. Train again in ML Lab.")
    try:
        probabilities = ml.predict_proba(bundle, body.model or bundle.context["best_model"], body.row)
        return {
            "model": body.model or bundle.context["best_model"],
            "labels": bundle.context["labels"],
            "probabilities": probabilities,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/python/answer")
def answer(body: AnswerBody):
    try:
        return answer_with_booking(
            body.question,
            body.rows,
            body.cols,
            body.profile,
            body.suggestions,
            body.trained,
            body.documents,
            body.booking,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/python/document")
def document(body: DocumentBody):
    try:
        data = base64.b64decode(body.content, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="That document could not be decoded.") from exc
    try:
        text, truncated = extract_text(body.name, data)
    except DocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"name": body.name, "text": text, "truncated": truncated}


@app.post("/api/python/transcribe")
def transcribe(body: TranscribeBody):
    """Turn a recorded voice note into text for the normal Converse pipeline."""
    try:
        audio = base64.b64decode(body.content, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="That voice note could not be decoded.") from exc
    if not audio:
        raise HTTPException(status_code=400, detail="That voice note was empty.")
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="That voice note is too long.")

    model_name, model = _asr_model()
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=_asr_suffix(body.name), delete=False) as handle:
            handle.write(audio)
            temp_path = handle.name
        segments, info = model.transcribe(
            temp_path,
            language=(body.language or None),
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return {
            "text": text,
            "language": getattr(info, "language", None),
            "duration": getattr(info, "duration", None),
            "model": model_name,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("VIDO_PYTHON_PORT", "8000")))
