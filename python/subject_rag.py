"""Pilot subject router for ViDo's Python answer engine.

The shared card store lives under ``python/subject_data``. Ten pilot subjects
keep 50 cards each, while expanded branch, book and research-paper catalogs add
broader coverage. Questions are classified before building a short answer.
The cards are plain text lines:

    title||aliases||summary||formula||points

This module is deliberately dependency-free and offline. Wikipedia stays the
fallback for topics outside the pilot set.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import subject_advanced
from wikipedia import strip_leading_fillers


DATA_DIR = Path(__file__).with_name("subject_data")

SUBJECT_NAMES: Dict[str, str] = {
    "math": "Mathematics",
    "physics": "Physics",
    "chemistry": "Chemistry",
    "biology": "Biology",
    "economics": "Economics",
    "computer_science": "Computer Science",
    "history": "History",
    "geography": "Geography",
    "politics_law": "Politics and Law",
    "psychology": "Psychology",
    "books": "Books and References",
    "research_papers": "Research Papers",
    "physics_branches": "Physics Branches",
    "math_branches": "Mathematics Branches",
    "statistics_branches": "Statistics Branches",
    "probability_statistics": "Probability and Statistics",
    "science_branches": "Science Branches",
    "computing_branches": "Computing Branches",
    "economics_branches": "Economics Branches",
    "humanities_branches": "Humanities Branches",
    "engineering_branches": "Engineering Branches",
    "health_branches": "Health Branches",
}

CORE_SUBJECT_KEYS = frozenset({
    "math", "physics", "chemistry", "biology", "economics",
    "computer_science", "history", "geography", "politics_law", "psychology",
})
EXPANDED_SUBJECT_KEYS = frozenset(
    key for key in SUBJECT_NAMES if key not in CORE_SUBJECT_KEYS
)

SUBJECT_SIGNALS: Dict[str, Tuple[str, ...]] = {
    "math": ("math", "maths", "mathematic", "algebra", "calculus", "geometry", "matrix", "probability", "statistic", "equation"),
    "physics": ("physics", "physic", "mechanic", "quantum", "relativity", "thermodynamic", "electric", "magnetic", "wave", "force"),
    "chemistry": ("chemistry", "chemical", "molecule", "atom", "reaction", "acid", "bond", "mole", "organic", "solution"),
    "biology": ("biology", "biological", "cell", "gene", "dna", "evolution", "organism", "ecosystem", "enzyme", "species"),
    "economics": ("economy", "economic", "market", "price", "demand", "supply", "inflation", "gdp", "trade", "tax"),
    "computer_science": ("computer", "computing", "programming", "software", "algorithm", "database", "network", "data", "code", "java"),
    "history": ("history", "historical", "ancient", "empire", "revolution", "war", "dynasty", "civilization", "medieval", "treaty"),
    "geography": ("geography", "geographic", "climate", "weather", "river", "mountain", "ocean", "continent", "map", "tectonic"),
    "politics_law": ("politic", "political", "law", "legal", "government", "constitution", "court", "election", "rights", "state"),
    "psychology": ("psychology", "psychological", "behavior", "behaviour", "memory", "emotion", "personality", "therapy", "conditioning", "cognitive"),
}

# These are task/UI requests, not academic topic questions. "graph theory" is a
# valid topic, but "process graph for this" is a flowchart follow-up.
NON_SUBJECT_QUESTION = re.compile(
    r"\b(?:process|flow)\s+graph\b|\bgraph\s+for\s+(?:this|that|it)\b|\bflowchart\b",
    re.I,
)

MIN_SCORE = 72.0
MAX_ANSWER_CHARS = 1200


@dataclass(frozen=True)
class TopicCard:
    subject_key: str
    subject: str
    title: str
    aliases: Tuple[str, ...]
    summary: str
    formula: str
    points: str


@dataclass(frozen=True)
class Classification:
    subject: str
    subject_key: str
    topic: str
    score: float
    matched_alias: str


_CARDS: Optional[List[TopicCard]] = None
_TOKEN_DF: Optional[Dict[str, float]] = None
_SINGLE_ALIASES: Optional[set] = None


# Source names are evidence metadata, never chat content. Card aliases may keep
# them for matching ("what is arxiv paper search" should still find the card),
# but every visible field is rewritten before it reaches an answer.
BADGE_TEXT = (
    (re.compile(r"\barxiv\b", re.I), "preprint archive"),
    (re.compile(r"\bwikipedia\b", re.I), "encyclopedia"),
)


def _badge_free(text: str) -> str:
    value = str(text or "")
    for pattern, replacement in BADGE_TEXT:
        value = pattern.sub(replacement, value)
    return value


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())).strip()


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("sses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(text: str) -> List[str]:
    return [_stem(word) for word in _normalize(text).split() if word]


def _token_set(text: str) -> set:
    return set(_tokens(text))


def _phrase_key(text: str) -> str:
    return " ".join(_tokens(text))


def _clean_question(question: str) -> str:
    value = strip_leading_fillers(question)
    value = re.sub(r"[?]+", " ", value)
    value = re.sub(r"\b(?:a|an|the)\b", " ", value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip()


def _topic_phrase(question: str) -> str:
    value = _clean_question(question)
    value = re.sub(
        r"^(?:tell me about|explain|describe|summari[sz]e|derive|analyze|analyse|"
        r"discuss|compare|contrast|evaluate|examine|explore|what about|about)\s+",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(r"^(?:why|how)\s+", "", value, flags=re.I)
    value = re.sub(
        r"\s+and\s+(?:describe|explain|discuss|state|show|derive|including)\b.*$",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"\s+(?:have|has|that|which|because|using|where|when)\b.*$",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"^(?:what|which|who|when|where|why|how)\s+"
        r"(?:is|are|was|were|do|does|did|can|could|should|would)\s+",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(r"^(?:is|are|what's)\s+", "", value, flags=re.I)
    # Keep the topic when the user asks for a deeper explanation. Phrases such
    # as "at an advanced level, including its formula and the main points to
    # remember" are framing, not part of the topic name.
    value = re.sub(r"\s*,?\s*including\b.*$", "", value, flags=re.I)
    value = re.sub(
        r"\s+at\s+(?:an?\s+)?(?:advanced|graduate|expert|intermediate|"
        r"introductory|basic|beginner|detailed|simpler?)\s+level\b.*$",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(r"\s+main points?\b.*$", "", value, flags=re.I)
    return value.strip(" .,")


def reload_cards() -> List[TopicCard]:
    """Reload and validate pilot cards. Useful for tests and data edits."""
    global _CARDS, _TOKEN_DF, _SINGLE_ALIASES
    cards: List[TopicCard] = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        subject_key = path.stem
        subject = SUBJECT_NAMES.get(subject_key, subject_key.replace("_", " ").title())
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fields = line.split("||")
            if len(fields) != 5:
                continue
            raw_title, aliases, summary, formula, points = (field.strip() for field in fields)
            if not raw_title or not aliases or not summary:
                continue
            title = _badge_free(raw_title)
            summary = _badge_free(summary)
            formula = _badge_free(formula)
            points = _badge_free(points)
            alias_list = tuple(alias.strip() for alias in aliases.split(",") if alias.strip())
            # Questions drop articles ("What is An Introduction to...?" becomes
            # "introduction to..."), so keep an article-free alias as well. Keep
            # the raw title for matching even when the displayed title is
            # badge-free (for example "arXiv Paper Search").
            title_aliases: List[str] = []
            for candidate in dict.fromkeys((raw_title, title)):
                title_aliases.append(candidate)
                article_free = re.sub(r"\b(?:a|an|the)\b", " ", candidate, flags=re.I)
                article_free = re.sub(r"\s+", " ", article_free).strip()
                if article_free and article_free != candidate:
                    title_aliases.append(article_free)
            cards.append(
                TopicCard(
                    subject_key=subject_key,
                    subject=subject,
                    title=title,
                    aliases=alias_list + tuple(title_aliases),
                    summary=summary,
                    formula=formula,
                    points=points,
                )
            )
    _CARDS = cards
    _TOKEN_DF = _build_idf(cards)
    # Only core one-word aliases act as "competing specific terms". Expanded
    # catalogs contain many generic single words (data, structure, book) that
    # would otherwise penalise a correct core answer for mentioning them.
    _SINGLE_ALIASES = {
        tokens[0]
        for card in cards
        if card.subject_key in CORE_SUBJECT_KEYS
        for alias in card.aliases
        for tokens in [_tokens(alias)]
        if len(tokens) == 1
    }
    return cards


def load_cards() -> List[TopicCard]:
    global _CARDS
    if _CARDS is None:
        reload_cards()
    return _CARDS or []


def _build_idf(cards: List[TopicCard]) -> Dict[str, float]:
    document_frequency: Counter = Counter()
    for card in cards:
        card_tokens = set()
        for alias in card.aliases:
            card_tokens.update(_tokens(alias))
        for token in card_tokens:
            document_frequency[token] += 1
    total = max(1, len(cards))
    return {
        token: math.log((total + 1) / (count + 1)) + 0.2
        for token, count in document_frequency.items()
    }


def _idf() -> Dict[str, float]:
    global _TOKEN_DF
    if _TOKEN_DF is None:
        reload_cards()
    return _TOKEN_DF or {}


def _single_aliases() -> set:
    global _SINGLE_ALIASES
    if _SINGLE_ALIASES is None:
        reload_cards()
    return _SINGLE_ALIASES or set()


def _subject_tokens(subject_key: str) -> set:
    return _token_set(" ".join(SUBJECT_SIGNALS.get(subject_key, ())))


def _score_card(
    card: TopicCard,
    clean_key: str,
    topic_key: str,
    raw_clean_key: str,
    raw_topic_key: str,
    topic_tokens: set,
    question_tokens: set,
    idf_map: Dict[str, float],
    single_aliases: set,
) -> Tuple[float, str]:
    best_score = 0.0
    best_alias = ""

    for alias in card.aliases:
        alias_key = _phrase_key(alias)
        alias_tokens = alias_key.split()
        if not alias_tokens:
            continue

        informativeness = sum(idf_map.get(token, 1.0) for token in alias_tokens) / len(alias_tokens)
        # A one-word alias such as "stack" is more specific than the broader
        # "data structure" it belongs to when both appear in the question.
        specificity = (4 - min(len(alias_tokens), 4)) * 1.5
        base = 55.0 + min(22.0, informativeness * 4.0) + specificity
        is_title_alias = alias_key == _phrase_key(card.title)

        if alias_key == topic_key or alias_key == clean_key:
            # An exact title match is the strongest signal. This keeps a topic
            # such as "Ideal gas equation" with its own chemistry card instead
            # of losing to the same phrase listed as an alias on a physics card.
            score = base + 45.0 + (6.0 if is_title_alias else 0.0)
        elif f" {alias_key} " in f" {clean_key} ":
            # Both sides are normalized, space-separated token strings. Plain
            # containment is equivalent to the old boundary regex and avoids
            # recompiling thousands of patterns on every question.
            score = base + 25.0
        elif set(alias_tokens) <= question_tokens:
            score = base + 18.0
        else:
            overlap = len(set(alias_tokens) & question_tokens)
            if len(alias_tokens) >= 2 and overlap >= max(1, math.ceil(len(alias_tokens) * 0.6)):
                score = base + 5.0 + overlap * 2.0
            else:
                continue

        if _normalize(alias) in (raw_topic_key, raw_clean_key):
            score += 4.0

        # A bare card title such as "Depression" should not beat a more
        # qualified synonym on the same card when the qualifier is absent.
        if is_title_alias:
            alias_word_set = set(alias_tokens)
            for other_alias in card.aliases:
                other_set = set(_tokens(other_alias))
                if alias_word_set and alias_word_set < other_set:
                    if other_set - alias_word_set - question_tokens:
                        score -= 10.0
                    break

        if len(alias_tokens) == 1 and len(question_tokens) >= 5 and idf_map.get(alias_tokens[0], 0.0) < 2.5:
            score -= 12.0

        # "nash equilibrium" is not "equilibrium", and "stack data structure"
        # is not answered by the broader "data structure" card. Unmatched
        # specific words are a stronger signal than shared generic words.
        unmatched = topic_tokens - set(alias_tokens)
        if unmatched:
            out_of_vocabulary = [token for token in unmatched if token not in idf_map]
            competing_aliases = [token for token in unmatched if token in single_aliases]
            modifiers = [
                token for token in unmatched
                if token in idf_map and token not in single_aliases
            ]
            score -= 45.0 * len(out_of_vocabulary)
            score -= 18.0 * len(competing_aliases)
            score -= 4.0 * len(modifiers)

        if score > best_score:
            best_score = score
            best_alias = alias

    if best_score <= 0:
        return 0.0, ""

    if question_tokens & _subject_tokens(card.subject_key):
        best_score += 8.0
    # Expanded catalogs deliberately overlap core topics (for example "data
    # structures"). A small preference keeps the original pilot answers stable
    # while still letting a new card win when no core card is a real match.
    if card.subject_key in EXPANDED_SUBJECT_KEYS:
        best_score -= 12.0
    best_score += min(len(_token_set(card.points) & question_tokens), 3) * 2.0
    return best_score, best_alias


def _rank_cards(question: str) -> List[Tuple[float, str, TopicCard]]:
    cards = load_cards()
    if not cards or NON_SUBJECT_QUESTION.search(question):
        return []

    cleaned = _clean_question(question)
    if len(cleaned) < 3:
        return []

    clean_key = _phrase_key(cleaned)
    topic_phrase = _topic_phrase(question)
    topic_key = _phrase_key(topic_phrase)
    raw_clean_key = _normalize(cleaned)
    raw_topic_key = _normalize(topic_phrase)
    topic_tokens = _token_set(topic_phrase)
    question_tokens = _token_set(cleaned)
    idf_map = _idf()
    single_aliases = _single_aliases()

    ranked: List[Tuple[float, str, TopicCard]] = []
    for card in cards:
        score, alias = _score_card(
            card,
            clean_key,
            topic_key,
            raw_clean_key,
            raw_topic_key,
            topic_tokens,
            question_tokens,
            idf_map,
            single_aliases,
        )
        if score >= MIN_SCORE:
            ranked.append((score, alias, card))
    ranked.sort(key=lambda item: (-item[0], item[2].subject_key, item[2].title))
    return ranked


def rank_cards(question: str) -> List[Tuple[float, str, TopicCard]]:
    """Public ranked card list so the shared concept graph can use the same scoring."""
    return _rank_cards(question)


def token_idf() -> Dict[str, float]:
    """Public IDF map for graph-based concept retrieval."""
    return dict(_idf())


def classify(question: str) -> Optional[Classification]:
    """Return the best pilot subject/topic for a question, if confident."""
    ranked = _rank_cards(question)
    if not ranked:
        return None
    score, alias, card = ranked[0]
    return Classification(
        subject=card.subject,
        subject_key=card.subject_key,
        topic=card.title,
        score=score,
        matched_alias=alias,
    )


def answer(question: str) -> Optional[str]:
    """Build a concise local answer from one pilot subject's topic cards."""
    ranked = _rank_cards(question)
    if not ranked:
        return None

    best_score, _, best_card = ranked[0]
    related = [card for score, _, card in ranked if card.subject_key == best_card.subject_key]

    if subject_advanced.is_advanced_request(question):
        other_related = [card for card in related if card is not best_card]
        return subject_advanced.build(best_card, other_related)

    selected: List[TopicCard] = []
    for card in [best_card] + related:
        if card not in selected:
            selected.append(card)
        if len(selected) == 2:
            break

    parts: List[str] = []
    for card in selected:
        chunk = f"{card.title}: {card.summary}"
        if card.formula:
            chunk += f" Quick form: {card.formula}"
        if card.points:
            chunk += f" Remember: {card.points}"
        parts.append(chunk)

    text = " ".join(parts).strip()
    return text[:MAX_ANSWER_CHARS].rstrip() + ("…" if len(text) > MAX_ANSWER_CHARS else "")


__all__ = [
    "Classification",
    "CORE_SUBJECT_KEYS",
    "EXPANDED_SUBJECT_KEYS",
    "TopicCard",
    "answer",
    "classify",
    "load_cards",
    "rank_cards",
    "reload_cards",
    "token_idf",
]
