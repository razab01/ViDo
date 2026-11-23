"""Wikipedia fallback for ViDo's Python answer engine.

Local world facts and the dataset graph answer first. When neither can answer,
this module searches Wikipedia and uses Hugging Face's fast tokenizer to choose
the page that best matches the question. The extract is returned as the answer,
so general-knowledge questions no longer fall through to "not enough data".
"""

from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Sequence

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "ViDo/1.0 (local answer engine; contact: local-user)"
TIMEOUT = 6
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE: Dict[str, str] = {}

_TOKENIZER = None
_TOKENIZER_TRIED = False


def _hf_tokens(text: str) -> List[str]:
    """Tokenize with the MiniLM Hugging Face tokenizer, with a regex fallback."""
    global _TOKENIZER, _TOKENIZER_TRIED
    if not _TOKENIZER_TRIED:
        _TOKENIZER_TRIED = True
        try:
            from tokenizers import Tokenizer

            _TOKENIZER = Tokenizer.from_pretrained(MODEL)
        except Exception:
            _TOKENIZER = None

    if _TOKENIZER is not None:
        try:
            return [
                token.lower()
                for token in _TOKENIZER.encode(str(text or "")).tokens
                if token and not token.startswith("[")
            ]
        except Exception:
            pass
    return [word.lower() for word in re.findall(r"[a-z0-9]+", str(text or "").lower())]


def topic_phrase(question: str) -> str:
    """Strip question wording so 'Tell me about India' becomes 'India'."""
    topic = re.sub(r"\?+\s*$", "", str(question or "").strip())
    topic = re.sub(
        r"^(?:tell me about|explain|describe|summari[sz]e|what about|about)\s+",
        "",
        topic,
        flags=re.I,
    )
    topic = re.sub(
        r"^(?:what|which|who|when|where|why|how)\s+"
        r"(?:is|are|was|were|do|does|did|can|could|should|would)\s+",
        "",
        topic,
        flags=re.I,
    )
    topic = re.sub(r"^(?:is|are|what's|what is)\s+", "", topic, flags=re.I)
    return topic.strip(" .,")


def _get_json(params: Dict[str, str]) -> Optional[dict]:
    url = API + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.load(response)
    except Exception:
        return None


def _search(topic: str, limit: int = 4) -> List[dict]:
    payload = _get_json(
        {
            "action": "query",
            "list": "search",
            "srsearch": topic,
            "srnamespace": "0",
            "srlimit": str(limit),
            "format": "json",
        }
    )
    if not payload:
        return []
    titles = [str(item.get("title") or "") for item in payload.get("query", {}).get("search", [])]
    titles = [title for title in titles if title][:limit]
    if not titles:
        return []

    payload = _get_json(
        {
            "action": "query",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "exsentences": "4",
            "redirects": "1",
            "titles": "|".join(titles),
            "format": "json",
        }
    )
    if not payload:
        return []
    candidates: List[dict] = []
    by_title = {title: index for index, title in enumerate(titles)}
    for page in payload.get("query", {}).get("pages", {}).values():
        extract = re.sub(r"\s+", " ", str(page.get("extract") or "")).strip()
        title = str(page.get("title") or "")
        if extract:
            candidates.append(
                {"title": title, "extract": extract, "rank": by_title.get(title, len(titles))}
            )
    return candidates


def _best_extract(question: str, topic: str, candidates: Sequence[dict]) -> Optional[str]:
    query_tokens = set(_hf_tokens(question)) | set(_hf_tokens(topic))
    if not query_tokens:
        return None

    best: Optional[str] = None
    best_score = 0.0
    for candidate in candidates:
        title_tokens = set(_hf_tokens(candidate["title"]))
        body_tokens = set(_hf_tokens(candidate["extract"][:1500]))
        score = 3 * len(query_tokens & title_tokens) + len(query_tokens & body_tokens)
        # Wikipedia's own relevance order matters, and an exact title match is
        # almost always the article the user meant.
        score += max(0, 5 - int(candidate.get("rank", 5))) * 3
        lowered = candidate["title"].lower()
        if lowered == topic.lower():
            score += 20
        elif lowered.endswith(" " + topic.lower()):
            score += 5
        score /= math.sqrt(len(body_tokens) + 1)
        if score > best_score:
            best_score = score
            best = candidate["extract"]
    return best


def _trim(text: str) -> str:
    if len(text) <= 700:
        return text
    cut = text.rfind(". ", 0, 700)
    if cut > 200:
        return text[: cut + 1]
    return text[:700].rstrip() + "…"


def answer(question: str) -> Optional[str]:
    """Return a Wikipedia-backed answer when local routing found nothing."""
    topic = topic_phrase(question)
    if len(topic) < 3 or len(topic.split()) > 8:
        return None
    key = topic.lower()
    if key not in CACHE:
        candidates = _search(topic)
        extract = _best_extract(question, topic, candidates)
        CACHE[key] = _trim(extract) if extract else ""
    return f"Wikipedia: {CACHE[key]}" if CACHE[key] else None
