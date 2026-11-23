"""FastAPI bridge between the React app and the Python ML/Graph-RAG stack."""

from __future__ import annotations

import os
import sys
from typing import Dict

# Keep a local math.py from shadowing Python's standard library module.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.append(HERE)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import ml
from graph_rag import answer as answer_with_graph
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
    trained: dict = {}


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
        return {
            "answer": answer_with_graph(
                body.question,
                body.rows,
                body.cols,
                body.profile,
                body.suggestions,
                body.trained,
            )
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("VIDO_PYTHON_PORT", "8000")))
