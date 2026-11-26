"""arXiv paper search for ViDo's Python answer engine.

Wikipedia remains the first external evidence source. arXiv is the second: it
is queried only when local cards, solvers and Wikipedia do not answer, or when
the question explicitly asks for papers, research or recent work. Results are
used as answer text and as internal evidence; no engine or source badge is
added to chat messages.

The implementation uses only the standard library so it stays offline-friendly:
network failures simply return no result.
"""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from wikipedia import strip_leading_fillers, topic_phrase


API = "https://export.arxiv.org/api/query"
USER_AGENT = "ViDo/1.0 (local answer engine; contact: local-user)"
TIMEOUT = 8
ATOM = "{http://www.w3.org/2005/Atom}"

CACHE: Dict[str, List[dict]] = {}
CACHE_TTL = 600


@dataclass(frozen=True)
class PaperEvidence:
    """Internal metadata for one paper. Never rendered as a chat badge."""

    id: str
    title: str
    published: str
    summary: str


@dataclass(frozen=True)
class Evidence:
    """Internal paper list supporting an arXiv-backed answer."""

    topic: str
    papers: Tuple[PaperEvidence, ...]


PaperSink = Callable[[Sequence[dict]], None]

PROCEDURAL_QUESTION = re.compile(
    r"^(?:how\s+to|how\s+do\s+i|how\s+can\s+i|steps?\s+to|guide\s+to|tutorial\s+on)\b",
    re.I,
)
RESEARCH_CUE = re.compile(
    r"\b(?:arxiv|paper|papers|research|study|studies|recent|survey|review|"
    r"preprint|literature|citation|citations|publication|published)\b",
    re.I,
)

CATEGORY_MAP: Dict[str, Tuple[str, ...]] = {
    "math": ("math.NT", "math.CO", "math.AG", "math.AP", "math.PR", "math.ST"),
    "math_branches": ("math.AG", "math.AP", "math.CO", "math.DG", "math.PR", "math.ST"),
    "statistics": ("stat.TH", "stat.ME", "stat.ML", "stat.AP", "stat.CO"),
    "statistics_branches": ("stat.TH", "stat.ME", "stat.ML", "stat.AP", "stat.CO"),
    "physics": ("quant-ph", "gr-qc", "hep-th", "hep-ph", "cond-mat.stat-mech", "astro-ph.CO"),
    "physics_branches": (
        "quant-ph", "gr-qc", "hep-th", "hep-ph", "nucl-th", "cond-mat.stat-mech",
        "astro-ph.CO", "physics.plasm-ph", "physics.flu-dyn", "physics.optics",
    ),
    "science_branches": ("physics.chem-ph", "physics.bio-ph", "q-bio.BM", "q-bio.PE", "physics.geo-ph"),
    "computing_branches": ("cs.LG", "cs.CL", "cs.CV", "cs.CR", "cs.DS", "cs.DB", "cs.AI"),
    "computer_science": ("cs.LG", "cs.CL", "cs.CV", "cs.CR", "cs.DS", "cs.DB", "cs.AI"),
    "economics_branches": ("econ.EM", "econ.GN", "econ.TH", "q-fin.ST", "q-fin.PM"),
    "economics": ("econ.EM", "econ.GN", "econ.TH"),
    "humanities_branches": ("cs.DL", "physics.hist-ph", "physics.soc-ph"),
    "engineering_branches": ("eess.SY", "eess.SP", "physics.acc-ph", "physics.ins-det"),
    "health_branches": ("q-bio.NC", "q-bio.PE", "stat.AP", "physics.med-ph"),
    "research_papers": ("cs.LG", "stat.ML", "quant-ph", "hep-th", "astro-ph.CO"),
    "books": ("math.HO", "physics.hist-ph", "cs.DL"),
}


def categories_for(subject_key: Optional[str]) -> Tuple[str, ...]:
    """Return arXiv categories for a classified subject key."""
    if not subject_key:
        return ()
    return CATEGORY_MAP.get(subject_key, ())


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _topic(question: str) -> str:
    topic = topic_phrase(strip_leading_fillers(str(question or "")))
    # "find recent papers on X" and "show me research about X" are requests
    # for papers, not part of the paper topic. Strip the instruction before
    # removing the research words themselves.
    topic = re.sub(
        r"^(?:(?:find|show|give|list|search(?:\s+for)?|look\s+for|suggest|recommend|share|get)\s+"
        r"(?:me\s+)?(?:some\s+|any\s+|the\s+)?"
        r")?"
        r"(?:(?:recent|latest|new|paper|papers|research|study|studies|survey|review)\s+)+",
        "",
        topic,
        flags=re.I,
    )
    topic = re.sub(
        r"^(?:on|about|for|of|regarding|concerning|related\s+to)\s+",
        "",
        topic,
        flags=re.I,
    )
    topic = re.sub(
        r"\b(?:arxiv|paper|papers|research|study|studies|recent|latest|preprint|literature)\b",
        " ",
        topic,
        flags=re.I,
    )
    return _clean(topic).strip(" .?!,")


def should_search(question: str, subject_key: Optional[str] = None) -> bool:
    """Search arXiv only for genuine research questions, not task requests."""
    text = strip_leading_fillers(str(question or "").strip())
    if not text or PROCEDURAL_QUESTION.search(text):
        return False
    return bool(subject_key or RESEARCH_CUE.search(text))


def search(topic: str, categories: Sequence[str] = (), limit: int = 3) -> List[dict]:
    """Return a small list of arXiv papers for a topic.

    Network and parsing failures return an empty list so the answer engine never
    becomes dependent on arXiv availability.
    """
    topic = _clean(topic)
    if len(topic) < 3:
        return []

    cats = [category for category in categories if category][:4]
    cache_key = f"{'|'.join(cats)}::{topic.lower()}::{limit}"
    cached = CACHE.get(cache_key)
    # An empty result list is a valid cache entry. Indexing cached[0] on it
    # used to raise IndexError on the next identical paper request.
    if cached and time.time() - cached[0]["_cached_at"] < CACHE_TTL:
        return [{key: value for key, value in paper.items() if not key.startswith("_")} for paper in cached]

    words = [word for word in re.findall(r"[A-Za-z0-9]+", topic) if len(word) > 1][:8]
    if not words:
        return []

    clauses: List[str] = []
    if cats:
        clauses.append("(" + " OR ".join(f"cat:{category}" for category in cats) + ")")
    clauses.append('all:"' + " ".join(words) + '"')

    params = {
        "search_query": " AND ".join(clauses),
        "start": "0",
        "max_results": str(max(1, min(limit, 5))),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"

    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = response.read()
        root = ET.fromstring(payload)
    except (urllib.error.URLError, urllib.error.HTTPError, ET.ParseError, OSError, ValueError):
        return []

    papers: List[dict] = []
    now = time.time()
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = _clean(entry.findtext(f"{ATOM}id", ""))
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id
        title = _clean(entry.findtext(f"{ATOM}title", ""))
        summary = _clean(entry.findtext(f"{ATOM}summary", ""))
        if not title or not summary:
            continue
        published = _clean(entry.findtext(f"{ATOM}published", ""))[:10]
        authors = [
            _clean(author.findtext(f"{ATOM}name", ""))
            for author in entry.findall(f"{ATOM}author")
        ]
        papers.append({
            "id": arxiv_id,
            "title": title,
            "summary": summary,
            "published": published,
            "authors": [author for author in authors if author],
            "_cached_at": now,
        })
        if len(papers) >= max(1, min(limit, 5)):
            break

    CACHE[cache_key] = papers
    return [{key: value for key, value in paper.items() if not key.startswith("_")} for paper in papers]


def _badge_free(text: str) -> str:
    """Keep paper wording out of the visible answer while preserving meaning."""
    value = _clean(text)
    value = re.sub(r"(?i)\barxiv\b", "preprint archive", value)
    value = re.sub(r"(?i)\bwikipedia\b", "encyclopedia", value)
    return value


def evidence(question: str, papers: Sequence[dict]) -> Evidence:
    """Build internal paper metadata; callers decide whether to display any."""
    return Evidence(
        topic=_topic(question),
        papers=tuple(
            PaperEvidence(
                id=str(paper.get("id", "")),
                title=_badge_free(paper.get("title", "")),
                published=str(paper.get("published", ""))[:10],
                summary=_badge_free(paper.get("summary", "")),
            )
            for paper in papers[:3]
        ),
    )


def answer(
    question: str,
    subject_key: Optional[str] = None,
    papers_out: Optional[List[dict]] = None,
) -> Optional[str]:
    """Return a short paper-backed answer when local sources found nothing."""
    if not should_search(question, subject_key):
        return None

    # Import lazily to avoid a circular import through graph_rag.
    if not subject_key:
        try:
            import subject_rag

            classification = subject_rag.classify(question)
            subject_key = classification.subject_key if classification else None
        except Exception:
            subject_key = None

    topic = _topic(question)
    if len(topic) < 3:
        return None

    papers = search(topic, categories_for(subject_key), limit=3)
    if not papers:
        return None
    if papers_out is not None:
        papers_out.extend(papers)

    lead = "Recent research" if RESEARCH_CUE.search(question) else "Relevant work"
    parts: List[str] = []
    for paper in papers[:2]:
        year = str(paper.get("published", ""))[:4] or "n.d."
        title = _badge_free(paper.get("title", "")) or "Untitled paper"
        summary = _badge_free(paper.get("summary", ""))
        if len(summary) > 180:
            summary = summary[:177].rstrip() + "..."
        paper_id = str(paper.get("id", "unknown")).strip() or "unknown"
        parts.append(f"{title} ({year}, {paper_id}): {summary}")

    text = f"{lead} on {topic}: " + " ".join(parts)
    return text[:900].rstrip() + ("…" if len(text) > 900 else "")


__all__ = [
    "CATEGORY_MAP",
    "Evidence",
    "PaperEvidence",
    "answer",
    "categories_for",
    "evidence",
    "search",
    "should_search",
]
