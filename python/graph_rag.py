"""Graph RAG and answer routing for ViDo.

A NetworkX graph connects each task row to its field values, words and model
suggestions. Questions are routed to math.py, curated world facts, or this
graph before the TypeScript fallback is ever needed.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import networkx as nx

import arxiv
import concept_graph
import plan_booking
import rag_python
import subject_advanced
from arxiv import answer as arxiv_answer
from smart_answers import answer as smart_answer
from smart_answers import GREETING_ANSWER, correct_typos, greeting_answer
from subject_advanced import is_advanced_request
from subject_rag import answer as subject_answer
from subject_rag import classify as classify_subject
from subject_rag import load_cards as load_subject_cards
from subject_solvers import solve as solve_subject_question
from vido_math import handle_query
from wikipedia import answer as wikipedia_answer
from wikipedia import strip_leading_fillers


EvidenceCallback = Callable[[str, str], None]
ArxivEvidenceCallback = Callable[[str, "arxiv.Evidence"], None]


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


def _wants_papers(question: str) -> bool:
    """True for explicit paper requests, not definition or study questions."""
    if not arxiv.RESEARCH_CUE.search(question):
        return False
    if re.search(r"\b(?:study|studies)\s+of\b", question, re.I):
        return False
    # "What is arXiv Paper Search?" and "Explain Preprint..." are concept
    # questions; the research cue is the topic, not a request for papers.
    if re.search(
        r"^(?:tell me about|explain|describe|what is|what are|who is|who was|what was)\b",
        question.strip(),
        re.I,
    ):
        return False
    return True


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
# Loaded document answers
# ---------------------------------------------------------------------------

def _document_chunks(text: str) -> List[str]:
    chunks: List[str] = []
    for paragraph in re.split(r"\n\s*\n+", str(text or "")):
        clean = re.sub(r"\s+", " ", paragraph).strip()
        if not clean:
            continue
        if len(clean) <= 1200:
            chunks.append(clean)
            continue
        for index in range(0, len(clean), 1000):
            piece = clean[index:index + 1200].strip()
            if piece:
                chunks.append(piece)
    return chunks


def _document_answer(question: str, documents: Optional[Sequence[dict]]) -> Optional[str]:
    """Answer from the document loaded in Chat Lab, before generic Wikipedia."""
    if not documents:
        return None

    wanted = set(tokens(question))
    wanted.update(token.lower() for token in re.findall(r"[A-Za-z]*-?\d+", question))
    if not wanted:
        return None

    hits: List[Tuple[int, int, str, str]] = []
    for document in documents:
        name = str(document.get("name") or "document")
        for index, chunk in enumerate(_document_chunks(document.get("text") or "")):
            chunk_tokens = set(tokens(chunk))
            score = sum(
                1 for token in wanted
                if token in chunk_tokens or re.search(rf"\b{re.escape(token)}\b", chunk, re.I)
            )
            if score:
                hits.append((score, index, name, chunk))

    if not hits:
        return None

    hits.sort(key=lambda item: (-item[0], item[1]))
    excerpts: List[str] = []
    for _, _, name, chunk in hits[:2]:
        excerpts.append(f'From "{name}": {chunk}')
    answer = " ".join(excerpts)
    return answer[:1600].rstrip() + ("…" if len(answer) > 1600 else "")


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
    # A bare "why" in an academic question must not become a task-data answer.
    if not (rows or cols or trained or suggestions):
        return None
    if "why" in low and not re.search(
        r"\btask|record|deadline|complet|dataset|row|column|feature|model\b",
        low,
    ):
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


def _find_column(cols: Sequence[str], patterns: Sequence[str]) -> Optional[str]:
    # Pattern order is intent order: "task_text" should beat "has_desc".
    for pattern in patterns:
        for col in cols:
            if re.search(pattern, col, re.I):
                return col
    return None


def _record_answer(question: str, rows: Sequence[dict], cols: Sequence[str]) -> Optional[str]:
    """Answer directly from row values instead of returning a vague graph hit."""
    low = question.lower()
    # Predictions belong to _likely_answer, not to records already completed.
    if re.search(r"\blikely|predict|chance|probability|estimate\b", low):
        return None

    id_col = _find_column(cols, [r"^(task[ _-]?)?id$"])
    title_col = (
        _find_column(cols, [r"task_text", r"title", r"desc", r"^text$", r"name"])
        or _title_column(cols)
    )
    done_col = _find_column(cols, [r"^complet", r"^done$", r"finished", r"^status$"])
    date_col = _find_column(cols, [r"due", r"deadline", r"date"])
    priority_col = _find_column(cols, [r"priority"])

    def label(row: dict) -> str:
        text = str(row.get(title_col) or "Untitled record")
        record_id = str(row.get(id_col) or "").strip() if id_col else ""
        return f"{record_id} — {text}" if record_id else text

    # Exact record lookup: "T09", "tell me about task T09".
    for token in re.findall(r"[A-Za-z]*-?\d+", question):
        if not id_col:
            break
        match = next(
            (row for row in rows if str(row.get(id_col, "")).strip().lower() == token.lower()),
            None,
        )
        if not match:
            continue
        parts = [f"{token} — {str(match.get(title_col) or 'Untitled record')}."]
        if done_col:
            parts.append(f"Status: {str(match.get(done_col) or 'unknown')}.")
        if date_col:
            parts.append(f"Due: {str(match.get(date_col) or 'not set')}.")
        if priority_col:
            parts.append(f"Priority: {str(match.get(priority_col) or 'not set')}.")
        return " ".join(parts)

    if re.search(r"completed|finished|done", low) and re.search(
        r"which|what|list|show|most|mostly|many|how much|are there|tasks|records", low
    ):
        finished = [
            row for row in rows
            if done_col and str(row.get(done_col, "")).strip().lower()
            in ("yes", "true", "1", "completed", "done", "finished", "completed on time")
        ]
        if re.search(r"how many|how much|count", low):
            return f"{len(finished)} {'task is' if len(finished) == 1 else 'tasks are'} marked completed in your data."
        if not finished:
            return "No tasks in your data are marked completed yet."
        titles = [label(row) for row in finished[:6]]
        return f"The completed {'task is' if len(finished) == 1 else 'tasks are'} {_quote_list(titles)}."

    if re.search(r"high priority|urgent", low) and priority_col:
        urgent = [row for row in rows if re.search(r"high|urgent", str(row.get(priority_col, "")), re.I)]
        if not urgent:
            return "No high-priority tasks are in your data."
        titles = [label(row) for row in urgent[:6]]
        return f"The high-priority {'task is' if len(urgent) == 1 else 'tasks are'} {_quote_list(titles)}."

    return None


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


def _advanced_hint_card(question: str):
    """Find the concept card behind an applied advanced question."""
    hint = subject_advanced.hint_for(question)
    if not hint:
        return None
    subject_key, title = hint
    return next(
        (
            card for card in load_subject_cards()
            if card.subject_key == subject_key and card.title.lower() == title.lower()
        ),
        None,
    )


def answer(
    question: str,
    rows: Sequence[dict],
    cols: Sequence[str],
    profile: Optional[dict] = None,
    suggestions: Optional[Sequence[dict]] = None,
    trained: Optional[dict] = None,
    documents: Optional[Sequence[dict]] = None,
    evidence_callback: Optional[EvidenceCallback] = None,
    arxiv_evidence_callback: Optional[ArxivEvidenceCallback] = None,
) -> str:
    raw_question = str(question or "").strip()

    # A bare greeting is a conversation turn, not an empty question. Answer it
    # warmly before filler stripping would turn it into nothing.
    greeting = greeting_answer(raw_question)
    if greeting:
        return greeting

    # "ok so what is X" is the same question as "what is X". Strip the lead-in
    # once here so routing, Wikipedia lookup and graph retrieval all agree.
    question = correct_typos(strip_leading_fillers(raw_question))
    if not question:
        return GREETING_ANSWER

    # Assigned before any paper lookup; declared here so the nested helper can
    # safely refer to it even if routing returns early.
    classification = None

    def with_evidence(text: str) -> str:
        """Let callers check Wikipedia in parallel without changing the answer."""
        if evidence_callback:
            try:
                evidence_callback(question, text)
            except Exception:
                pass
        return text

    def paper_answer() -> Optional[str]:
        """Return badge-free paper text and keep paper metadata internal."""
        details: List[dict] = []
        text = arxiv_answer(
            question,
            classification.subject_key if classification else None,
            details,
        )
        if not text:
            return None
        if arxiv_evidence_callback and details:
            try:
                arxiv_evidence_callback(question, arxiv.evidence(question, details))
            except Exception:
                pass
        # The Wikipedia callback still runs, so external evidence checks stay
        # parallel and internal even when papers provide the visible answer.
        return with_evidence(text)

    # A loaded document is active chat context, so it beats generic answers.
    doc_answer = _document_answer(question, documents)
    if doc_answer:
        return doc_answer

    # An explicit paper request is a source-type request, not a concept lookup.
    # It must win before the concept graph can claim a generic word such as
    # "graph" from the question.
    if _wants_papers(question):
        classification = classify_subject(question)
        arxiv_text = paper_answer()
        if arxiv_text:
            return arxiv_text

    # Numeric physics questions are owned by exact computation even when they
    # carry advanced framing. Check them before the advanced explanation route
    # can return a generic card instead of the calculated result.
    if not is_advanced_request(question):
        rag_physics = rag_python.physics_answer(question)
        if rag_physics:
            return with_evidence(rag_physics)

    # Advanced academic questions need the full explanation before a short
    # curated answer (or a simple maths handler) can claim the same keyword.
    if is_advanced_request(question):
        # Applied questions ("pH of 0.01 M HCl") often fail strict card ranking
        # and let an incidental word such as "limit" steal the classification.
        # A narrow topic hint is stronger evidence, so try it first.
        hint_card = _advanced_hint_card(question)
        if hint_card:
            deeper = subject_advanced.build(hint_card, [])
            if hint_card.subject_key.startswith("physics"):
                # Python still owns any numeric result; the card only supplies
                # the requested depth. This keeps advanced numeric questions
                # exact without letting a generic solver guess the concept.
                physics_exact = rag_python.physics_answer(question)
                if physics_exact:
                    return with_evidence(f"{physics_exact}\n\n{deeper}")
                return with_evidence(deeper)
            advanced_solver = solve_subject_question(question, None)
            if advanced_solver:
                return with_evidence(f"{advanced_solver}\n\n{deeper}")
            advanced_math = handle_query(question)
            if advanced_math:
                return with_evidence(f"{advanced_math}\n\n{deeper}")
            # Some applied concepts have no numeric payload at all. The
            # expanded card is still the correct advanced answer.
            return with_evidence(deeper)

        advanced_classification = classify_subject(question)
        if advanced_classification:
            advanced_solved = solve_subject_question(question, advanced_classification)
            if advanced_solved:
                return with_evidence(advanced_solved)
            advanced_answer = subject_answer(question)
            if advanced_answer:
                return with_evidence(advanced_answer)

    # The retrieval layer first understands which computation is being asked
    # for; Python executes the exact calculation and the same layer replies.
    rag_planned = rag_python.answer(question)
    if rag_planned:
        return rag_planned

    # Applied physics questions such as Debye length or numerical heat schemes
    # often resist a full parse but still have one exact governing relation.
    # Answer them here, after advanced card routes have had their chance.
    if not is_advanced_request(question):
        physics_concept = rag_python.physics_concept_answer(question)
        if physics_concept:
            return physics_concept

    math_answer = handle_query(question)
    if math_answer:
        return math_answer

    # Answer the question the person meant before searching Wikipedia for its
    # last few words. This fixes "capital of India", "NCR vs NCT", typos and
    # subjective follow-ups without touching dataset routing.
    direct_answer = smart_answer(question)
    if direct_answer:
        return direct_answer

    world_answer = _world_fact_answer(question)
    if world_answer:
        return world_answer

    # Phase 2: classify first, then let Math/Physics/Chemistry solvers handle
    # applied questions before the definition cards answer concept questions.
    # A callback can check Wikipedia in parallel; it never edits this text.
    classification = classify_subject(question)
    solved = solve_subject_question(question, classification)
    if solved:
        return with_evidence(solved)

    # One shared concept graph with subdomain partitions handles explanations,
    # comparisons, relationships and why/how questions from the same card store.
    # Deterministic handlers above still own maths, tasks and exact records.
    concept_answer = concept_graph.answer(question)
    if concept_answer:
        return with_evidence(concept_answer)

    # Phase 1 pilot: answer in-scope academic topics from the subject cards.
    # High-confidence cards stay offline; Wikipedia still handles everything
    # outside the ten pilot subjects.
    subject_answer_text = subject_answer(question)
    if subject_answer_text:
        return with_evidence(subject_answer_text)

    # Explicit general-knowledge questions should not be answered from an
    # incidental dataset keyword match. Search Wikipedia through HF first,
    # unless the person explicitly asked for papers or research.
    if _is_general_knowledge(question) and not _wants_papers(question):
        wiki_answer = wikipedia_answer(question)
        if wiki_answer:
            return with_evidence(wiki_answer)

    suggestions = list(suggestions or [])
    trained = trained or {}
    graph = build_graph(rows, cols, suggestions)

    # Specific question shapes win over the broad "likely records" answer:
    # a compare question mentions `completed` and a pattern question mentions
    # `predict`, but neither is asking for a ranked record list.
    candidate_answers = [
        _record_answer(question, rows, cols),
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

    # Explicit research questions prefer paper results over a generic
    # encyclopedia fallback. Wikipedia evidence is still checked internally by
    # with_evidence, and paper metadata stays in its own callback.
    if _wants_papers(question):
        arxiv_text = paper_answer()
        if arxiv_text:
            return arxiv_text

    # Last resort: anything not answered locally still gets a Wikipedia search.
    wiki_answer = wikipedia_answer(question)
    if wiki_answer:
        return with_evidence(wiki_answer)

    # Second internal evidence source. arXiv is consulted after local cards,
    # solvers and Wikipedia have all declined for non-research questions.
    arxiv_text = paper_answer()
    if arxiv_text:
        return arxiv_text

    return (
        "I understand the question, but I do not have enough information in this dataset yet. "
        "Upload a labelled file or ask a maths, general-knowledge or task question and I will answer directly."
    )


def answer_with_booking(
    question: str,
    rows: Sequence[dict],
    cols: Sequence[str],
    profile: Optional[dict] = None,
    suggestions: Optional[Sequence[dict]] = None,
    trained: Optional[dict] = None,
    documents: Optional[Sequence[dict]] = None,
    booking: Optional[dict] = None,
    evidence_callback: Optional[EvidenceCallback] = None,
    arxiv_evidence_callback: Optional[ArxivEvidenceCallback] = None,
) -> Dict[str, Any]:
    """Graph-RAG answer with an optional structured plan-booking action.

    The reply text stays identical to `answer()` for every ordinary question.
    Booking turns are owned by the same graph-RAG layer: it reads the active
    document, plans the tasks, asks one clarification when needed, and hands
    complete tasks back for the UI to create through its normal task callback.
    """
    result = plan_booking.plan(question, documents, booking)
    if result is not None:
        return result.as_response()

    text = answer(
        question,
        rows,
        cols,
        profile,
        suggestions,
        trained,
        documents,
        evidence_callback,
        arxiv_evidence_callback,
    )
    return {"answer": text}
