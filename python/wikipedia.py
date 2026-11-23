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
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Sequence

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "ViDo/1.0 (local answer engine; contact: local-user)"
TIMEOUT = 6
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE: Dict[str, str] = {}
# Disambiguation stubs such as "President of Spain may refer to:" are not
# answers. They are filtered out before ranking so chat replies stay useful.
DISAMBIGUATION = re.compile(r"\b(?:may refer to|could refer to|disambiguation|redirects here)\b", re.I)
FILLER_PREFIX = re.compile(
    r"^(?:ok(?:ay)?|so|and|now|well|yeah|yes|but|please|also)\b[\s,!.]*",
    re.I,
)
COMMON_TYPOS = (
    (re.compile(r"\boart\b", re.I), "part"),
    (re.compile(r"\bthus\b", re.I), "this"),
)
# A few common terms have a disambiguation page with the exact title. Prefer
# the article that matches the meaning people usually ask about.
TOPIC_ALIASES = {"blue chip": "Blue chip (stock market)"}

_TOKENIZER = None
_TOKENIZER_TRIED = False
_JSON_CACHE: Dict[str, dict] = {}
_LAST_REQUEST_AT = 0.0
MIN_REQUEST_INTERVAL = 1.2


def strip_leading_fillers(question: str) -> str:
    """Remove conversational lead-ins and fix common typos before routing."""
    value = str(question or "").strip()
    while True:
        updated = FILLER_PREFIX.sub("", value).lstrip()
        if updated == value:
            break
        value = updated
    for pattern, replacement in COMMON_TYPOS:
        value = pattern.sub(replacement, value)
    return value


def _normalize_title(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())).strip()


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
    """Strip question wording so 'ok so what is Nash equilibrium' becomes 'Nash equilibrium'."""
    topic = re.sub(r"\?+\s*$", "", strip_leading_fillers(question))
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
    """Fetch Wikipedia JSON with light throttling and 429 retry/backoff."""
    global _LAST_REQUEST_AT
    url = API + "?" + urllib.parse.urlencode(params)
    if url in _JSON_CACHE:
        return _JSON_CACHE[url]

    for attempt in range(2):
        wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _LAST_REQUEST_AT)
        if wait > 0:
            time.sleep(wait)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                payload = json.load(response)
            _LAST_REQUEST_AT = time.monotonic()
            _JSON_CACHE[url] = payload
            return payload
        except urllib.error.HTTPError as exc:
            _LAST_REQUEST_AT = time.monotonic()
            if exc.code == 429 and attempt == 0:
                try:
                    delay = min(20, max(1, int(exc.headers.get("Retry-After", "2"))))
                except (TypeError, ValueError):
                    delay = 2
                time.sleep(delay)
                continue
            return None
        except Exception:
            _LAST_REQUEST_AT = time.monotonic()
            return None
    return None


def _search(topic: str, limit: int = 10) -> List[dict]:
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
        if (
            extract
            and not DISAMBIGUATION.search(extract)
            and not DISAMBIGUATION.search(title)
            and len(extract) >= 60
        ):
            candidates.append(
                {"title": title, "extract": extract, "rank": by_title.get(title, len(titles))}
            )
    return candidates


def _best_extract(question: str, topic: str, candidates: Sequence[dict]) -> Optional[str]:
    topic_key = _normalize_title(topic)
    preferred_key = _normalize_title(TOPIC_ALIASES.get(topic_key, ""))
    query_tokens = set(_hf_tokens(question)) | set(_hf_tokens(topic))
    if not query_tokens:
        return None

    # When Wikipedia has the exact article, it beats shorter parenthetical
    # pages such as "Photosynthesis (board game)". Disambiguation pages were
    # already filtered out by _search, so this is the meaning people asked for.
    for candidate in candidates:
        candidate_key = _normalize_title(candidate["title"])
        if (topic_key and candidate_key == topic_key) or (
            preferred_key and candidate_key == preferred_key
        ):
            return candidate["extract"]

    best: Optional[str] = None
    best_score = float("-inf")
    for candidate in candidates:
        title = candidate["title"]
        extract = candidate["extract"]
        lowered = title.lower()
        title_key = _normalize_title(title)
        title_tokens = set(_hf_tokens(title))
        body_tokens = set(_hf_tokens(extract[:1500]))

        score = 4 * len(query_tokens & title_tokens) + len(query_tokens & body_tokens)
        score += max(0, 10 - int(candidate.get("rank", 10))) * 2

        exact = bool(topic_key) and title_key == topic_key
        preferred = bool(preferred_key) and title_key == preferred_key
        parenthetical = bool(topic_key) and lowered.startswith(topic_key + " (")
        if exact:
            score += 50
        elif preferred:
            score += 60
        elif parenthetical:
            score += 30
        elif topic_key and title_key.endswith(" " + topic_key):
            score += 8

        # A page that defines the topic in its opening sentence is a better
        # answer than a company or film that merely contains the same words.
        if topic_key and re.search(rf"\b{re.escape(topic_key)}\b", _normalize_title(extract[:180])):
            score += 10
        if not (exact or preferred or parenthetical):
            if topic_key and title_key.startswith(topic_key + " "):
                score -= 10
            if re.search(r"\b(?:film|album|company|stamps)\b", title_key) or re.search(
                r"\b(?:started as|was an American|released in|released on)\b",
                extract[:300],
                re.I,
            ):
                score -= 8

        score /= math.sqrt(len(body_tokens) + 1)
        if score > best_score:
            best_score = score
            best = extract
    return best


def _trim(text: str) -> str:
    if len(text) <= 700:
        return text
    cut = text.rfind(". ", 0, 700)
    if cut > 200:
        return text[: cut + 1]
    return text[:700].rstrip() + "…"


def _search_topics(question: str, topic: str) -> List[str]:
    """Try the best-known article first, then the literal topic and role rephrasings."""
    preferred = TOPIC_ALIASES.get(_normalize_title(topic), topic)
    topics = [preferred]
    if preferred != topic:
        topics.append(topic)
    match = re.search(
        r"\b(president|prime minister|king|queen|chancellor)\s+of\s+([a-z][\w\s-]+)",
        str(question or "").lower(),
    )
    if match:
        role, country = match.group(1), match.group(2).strip(" .,")
        if role == "president":
            topics.append(f"head of state of {country}")
        elif role in ("prime minister", "chancellor"):
            topics.append(f"head of government of {country}")
    return [value for index, value in enumerate(topics) if value and value not in topics[:index]]


def answer(question: str) -> Optional[str]:
    """Return a Wikipedia-backed answer when local routing found nothing."""
    topic = topic_phrase(question)
    if len(topic) < 3 or len(topic.split()) > 8:
        return None
    key = topic.lower()
    if key not in CACHE:
        extract = ""
        for candidate_topic in _search_topics(question, topic):
            candidates = _search(candidate_topic)
            extract = _best_extract(question, topic, candidates)
            if extract:
                break
        CACHE[key] = _trim(extract) if extract else ""
    return f"Wikipedia: {CACHE[key]}" if CACHE[key] else None
