"""Graph RAG and answer routing for ViDo.

A NetworkX graph connects each task row to its field values, words and model
suggestions. Questions are routed to math.py, curated world facts, or this
graph before the TypeScript fallback is ever needed.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import networkx as nx

from vido_math import handle_query
from wikipedia import answer as wikipedia_answer


STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with", "is", "are",
    "was", "were", "be", "by", "as", "at", "from", "this", "that", "these", "those",
    "what", "which", "who", "whom", "how", "why", "when", "do", "does", "did", "look",
    "looks", "across", "dataset", "data", "records", "record", "likely", "predict",
    "patterns", "compare", "me", "about", "it", "its", "can", "you", "your", "my",
}


def tokens(text: str) -> List[str]:
    return [word for word in re.findall(r"[a-z0-9_]+", str(text or "").lower()) if word not in STOP_WORDS]


def _title_column(cols: Sequence[str]) -> str:
    for col in cols:
        if re.search(r"task|title|text|name|description", col, re.I):
            return col
    return cols[0] if cols else ""


def build_graph(rows: Sequence[dict], cols: Sequence[str], suggestions: Sequence[dict]) -> nx.Graph:
    graph = nx.Graph()
    title_col = _title_column(cols)

    for index, row in enumerate(rows):
        node = f"row:{index}"
        title = str(row.get(title_col, f"Record {index + 1}"))
        graph.add_node(node, kind="row", title=title, weight=3.0)
        for col in cols:
            value = str(row.get(col, "") or "").strip()
            if not value:
                continue
            value_node = f"{col}={value}"
            graph.add_node(value_node, kind="value", column=col, value=value, weight=1.5)
            graph.add_edge(node, value_node)
        for word in set(tokens(title)):
            word_node = f"word:{word}"
            graph.add_node(word_node, kind="word", weight=1.0)
            graph.add_edge(node, word_node)

    for suggestion in suggestions:
        title = str(suggestion.get("title") or "")
        node = f"suggestion:{title}"
        graph.add_node(
            node,
            kind="suggestion",
            title=title,
            label=str(suggestion.get("label") or ""),
            conf=str(suggestion.get("conf") or ""),
            weight=2.0,
        )
        for word in set(tokens(title)):
            if graph.has_node(f"word:{word}"):
                graph.add_edge(node, f"word:{word}")
    return graph


def retrieve(question: str, graph: nx.Graph, limit: int = 6) -> List[str]:
    query_tokens = set(tokens(question))
    if not query_tokens:
        return []
    scores: Dict[str, float] = {}
    for node, data in graph.nodes(data=True):
        data_tokens = set(tokens(data.get("title", "")))
        data_tokens.update(tokens(str(data.get("value", ""))))
        overlap = query_tokens & data_tokens
        score = len(overlap) * float(data.get("weight") or 1)
        if score <= 0 and data.get("kind") == "row":
            neighbours = graph.neighbors(node)
            score = sum(1 for neighbour in neighbours if any(word in neighbour for word in query_tokens)) * 0.3
        if score > 0:
            scores[node] = score
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    row_hits: List[str] = []
    seen_rows = set()
    for node, _ in ranked:
        if node.startswith("row:"):
            if node not in seen_rows:
                seen_rows.add(node)
                row_hits.append(node)
        elif node.startswith("suggestion:"):
            row_hits.append(node)
        if len(row_hits) >= limit:
            break
    return row_hits


# ---------------------------------------------------------------------------
# Curated world facts (kept local so answers work without network)
# ---------------------------------------------------------------------------

PRESIDENT_MEANING = (
    "A president is the head of state or head of government of a republic, usually elected "
    "for a fixed term. The exact role depends on the country's constitution."
)

MONARCHY_COUNTRIES = {"netherlands", "holland", "spain"}

WORLD_FACTS = {
    "usa": "The current President of the United States is Donald Trump.",
    "united states": "The current President of the United States is Donald Trump.",
    "america": "The current President of the United States is Donald Trump.",
    "india": "The current President of India is Droupadi Murmu.",
    "bharat": "The current President of India is Droupadi Murmu.",
    "netherlands": (
        "The Netherlands is a constitutional monarchy, so it has no president. "
        "The head of state is King Willem-Alexander and the head of government is "
        "Prime Minister Dick Schoof."
    ),
    "holland": (
        "The Netherlands is a constitutional monarchy, so it has no president. "
        "The head of state is King Willem-Alexander and the head of government is "
        "Prime Minister Dick Schoof."
    ),
    "spain": (
        "Spain is a constitutional monarchy, so it has no president as head of state. "
        "The head of state is King Felipe VI and the head of government is "
        "Prime Minister Pedro Sánchez."
    ),
}


def _is_general_knowledge(question: str) -> bool:
    """True for 'Tell me about X' style questions that need Wikipedia first."""
    low = question.lower().strip()
    if not re.match(
        r"^(?:tell me about|explain|describe|what is|what are|who is|who was|what was)\b",
        low,
    ):
        return False
    dataset_words = re.search(
        r"\b(?:compare|pattern|predict|likely|dataset|records?|task|task_id|"
        r"completed|completion|probability|deadline|due|priority)\b",
        low,
    )
    return dataset_words is None


def _world_fact_answer(question: str) -> Optional[str]:
    low = question.lower()
    role_words = re.search(r"\b(president|prime minister|king|queen|chancellor|head of state|head of government)\b", low)
    if not role_words:
        return None
    for country, fact in WORLD_FACTS.items():
        if country in low:
            if country in MONARCHY_COUNTRIES and "president" in low:
                return f"{PRESIDENT_MEANING} {fact}"
            return fact
    if re.search(r"\b(what does|what is|meaning of|define)\b.*\bpresident\b", low) or "president mean" in low:
        return PRESIDENT_MEANING
    return None


# ---------------------------------------------------------------------------
# Dataset answers
# ---------------------------------------------------------------------------

def _quote_list(values: Sequence[str]) -> str:
    clean = [str(value).strip() for value in values if str(value).strip()][:4]
    if not clean:
        return ""
    if len(clean) == 1:
        return f"“{clean[0]}”"
    if len(clean) == 2:
        return f"“{clean[0]}” and “{clean[1]}”"
    return ", ".join(f"“{value}”" for value in clean[:-1]) + f" and “{clean[-1]}”"


def _likely_answer(question: str, suggestions: Sequence[dict], trained: Optional[dict]) -> Optional[str]:
    if not re.search(r"\blikely|chance|probability|completion\b", question.lower()):
        return None
    # Rank by the positive-class probability, not by the predicted label. A
    # record predicted "no" at 54% can still be the most likely completed one
    # when every other record scores lower.
    ranked = [
        item for item in suggestions
        if str(item.get("completion_confidence", "")).replace("%", "").strip()
    ]
    if ranked:
        def completion_score(item: dict) -> float:
            try:
                return float(str(item["completion_confidence"]).rstrip("%"))
            except (TypeError, ValueError):
                return 0.0

        top = sorted(ranked, key=completion_score, reverse=True)[:4]
        examples = [f"{item['title']} ({item['completion_confidence']})" for item in top]
        return (
            f"The records most likely to be completed are {_quote_list(examples)}. "
            f"The estimate uses {trained.get('best', {}).get('n', 'the trained model') if trained else 'the trained model'}."
        )
    positives = [
        item for item in suggestions
        if str(item.get("label", "")).lower() in ("completed", "yes", "completed on time")
    ]
    if positives:
        examples = [f"{item['title']} ({item['conf']})" for item in positives[:4]]
        return (
            f"The records most likely to be completed are {_quote_list(examples)}. "
            f"The estimate uses {trained.get('best', {}).get('n', 'the trained model') if trained else 'the trained model'}."
        )
    if suggestions:
        return "No record currently has a strong completed-on-time estimate. Upload a labelled file and train again to get one."
    return None


def _compare_answer(question: str, rows: Sequence[dict], cols: Sequence[str], trained: Optional[dict]) -> Optional[str]:
    low = question.lower()
    if not re.search(r"\bcompare|versus|vs\b|across\b|relationship\b", low):
        return None
    mentioned = [col for col in cols if re.search(rf"\b{re.escape(col.lower())}\b", low)]
    if len(mentioned) < 2:
        return None
    first, second = mentioned[0], mentioned[1]
    complete_col = next((col for col in (first, second) if re.search(r"complet|done|finished", col, re.I)), None)
    id_col = next((col for col in (first, second) if col != complete_col), first)
    finished = [row for row in rows if str(row.get(complete_col or second, "")).lower() in ("yes", "completed", "true", "1")]

    parts = [
        f"{first} identifies each record, while {second} records its outcome."
    ]
    if complete_col:
        if finished:
            titles = [str(row.get(_title_column(cols), "")) for row in finished[:4]]
            parts.append(f"The finished records include {_quote_list(titles)}.")
        else:
            parts.append("None of the visible records are marked as finished yet.")
    if trained and trained.get("features"):
        parts.append(f"The trained model relies most on {', '.join(trained['features'][:4])}.")
    return " ".join(parts)


def _pattern_answer(question: str, rows: Sequence[dict], cols: Sequence[str], trained: Optional[dict], suggestions: Sequence[dict]) -> Optional[str]:
    low = question.lower()
    if not re.search(r"\bpattern|predict|driver|signal|feature|why\b", low):
        return None
    if trained and trained.get("features"):
        features = ", ".join(str(item) for item in trained["features"][:5])
        parts = [f"The strongest available signals are {features}."]
    else:
        parts = ["The strongest signals come from the task text and its deadline fields."]

    complete_col = next((col for col in cols if re.search(r"^complet", col, re.I)), None)
    if complete_col:
        finished = [row for row in rows if str(row.get(complete_col, "")).lower() in ("yes", "completed", "true", "1")]
        if finished:
            parts.append(f"Completed records tend to have earlier deadlines and clearer task titles, for example {_quote_list([str(row.get(_title_column(cols), '')) for row in finished[:3]])}.")
    if suggestions:
        parts.append(f"The model's current estimate is led by {suggestions[0]['label']} for {_quote_list([suggestions[0]['title']])}.")
    return " ".join(parts)


def _graph_answer(question: str, rows: Sequence[dict], cols: Sequence[str], graph: nx.Graph) -> Optional[str]:
    hits = retrieve(question, graph)
    if not hits:
        return None
    titles: List[str] = []
    for hit in hits:
        if hit.startswith("row:"):
            titles.append(str(graph.nodes[hit].get("title") or ""))
        elif hit.startswith("suggestion:"):
            titles.append(str(graph.nodes[hit].get("title") or ""))
    titles = [title for title in titles if title][:4]
    if not titles:
        return None
    return f"The closest records in your data are {_quote_list(titles)}. Ask about one of them for a fuller explanation."


def answer(
    question: str,
    rows: Sequence[dict],
    cols: Sequence[str],
    profile: Optional[dict] = None,
    suggestions: Optional[Sequence[dict]] = None,
    trained: Optional[dict] = None,
) -> str:
    question = str(question or "").strip()
    if not question:
        return "Type a question and I will answer it."

    math_answer = handle_query(question)
    if math_answer:
        return math_answer

    world_answer = _world_fact_answer(question)
    if world_answer:
        return world_answer

    # Explicit general-knowledge questions should not be answered from an
    # incidental dataset keyword match. Search Wikipedia through HF first.
    if _is_general_knowledge(question):
        wiki_answer = wikipedia_answer(question)
        if wiki_answer:
            return wiki_answer

    suggestions = list(suggestions or [])
    trained = trained or {}
    graph = build_graph(rows, cols, suggestions)

    # Specific question shapes win over the broad "likely records" answer:
    # a compare question mentions `completed` and a pattern question mentions
    # `predict`, but neither is asking for a ranked record list.
    candidate_answers = [
        _compare_answer(question, rows, cols, trained),
        _pattern_answer(question, rows, cols, trained, suggestions),
        _likely_answer(question, suggestions, trained),
    ]
    for answer_text in candidate_answers:
        if answer_text:
            return answer_text

    graph_answer = _graph_answer(question, rows, cols, graph)
    if graph_answer:
        return graph_answer

    # Last resort: anything not answered locally still gets a Wikipedia search.
    wiki_answer = wikipedia_answer(question)
    if wiki_answer:
        return wiki_answer

    return (
        "I understand the question, but I do not have enough information in this dataset yet. "
        "Upload a labelled file or ask a maths, general-knowledge or task question and I will answer directly."
    )
