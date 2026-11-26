"""One shared concept graph with subdomain partitions.

Every card in ``subject_data`` becomes a concept node in a single graph. Nodes
keep a ``domain`` attribute (the subject partition) and their aliases, summary,
formula and points. Edges are built from shared aliases and shared informative
language, so a question can travel from one concept to a related one even when
they live in different subject partitions.

This module is the concept-level reasoning layer. Deterministic handlers for
exact maths, tasks, dates and documents still run before it. Nothing here calls
the network; Wikipedia remains an internal evidence check owned by the router.
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx

import subject_rag


CONCEPT_KIND = "concept"
MAX_SEEDS = 3
MAX_NEIGHBOURS = 4
MAX_TOKEN_POSTINGS = 40
MAX_ALIAS_POSTINGS = 14
MAX_ANSWER_CHARS = 1400

COMPARE_RE = re.compile(
    r"\b(?:compare|contrast|versus|vs\.?|difference(?:s)?\s+between|distinguish)\b"
    r"|\bhow\s+(?:are|do|is)\b.{0,60}\bdiffer\b",
    re.I,
)
RELATE_RE = re.compile(
    r"\b(?:relat(?:e|es|ed|ion|ionship)|connect(?:s|ed|ion)?|link(?:s|ed)?|connection)\b"
    r"|\bbetween\b.{0,80}\band\b",
    re.I,
)
MECHANISM_RE = re.compile(
    r"^\s*(?:why|how)\b"
    r"|\b(?:what causes?|cause of|reason for|mechanism of|how does .{1,60} work)\b",
    re.I,
)

# Words that are common in study questions but say nothing about a concept.
QUERY_STOP = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with",
    "is", "are", "was", "were", "be", "by", "as", "at", "from", "this",
    "that", "these", "those", "what", "which", "who", "why", "how", "when",
    "where", "do", "does", "did", "can", "could", "should", "would", "tell",
    "me", "about", "explain", "describe", "compare", "versus", "difference",
    "between", "relate", "relationship", "connect", "connection", "link",
    "mean", "means", "please", "also", "give", "show", "want", "know",
    "doe", "not", "find", "recent", "paper", "papers", "get", "term",
    "concept", "idea", "called", "known", "affect", "effect", "cause",
    "reason", "work", "works", "looking", "look", "search", "need",
}


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


def _clip(text: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    value = value.rstrip(".")
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _looks_like_formula(text: str) -> bool:
    return bool(re.search(r"[=≈→<>∫√^]|(?:\d)", str(text or "")))


def _line(card: subject_rag.TopicCard) -> str:
    """One grounded line of evidence for a concept node."""
    domain = subject_rag.SUBJECT_NAMES.get(card.subject_key, card.subject_key)
    return f"{card.title} ({domain}): {_clip(card.summary, 220)}"


_graph_cache: Optional[nx.Graph] = None
_known_vocabulary: Optional[Set[str]] = None


def _known_tokens() -> Set[str]:
    """Every token that appears in a concept title or alias."""
    global _known_vocabulary
    if _known_vocabulary is None:
        vocabulary: Set[str] = set()
        for card in subject_rag.load_cards():
            vocabulary.update(_tokens(" ".join([card.title, *card.aliases])))
        _known_vocabulary = vocabulary
    return _known_vocabulary


def _add_edge(
    graph: nx.Graph,
    left: str,
    right: str,
    weight: float,
    shared: Iterable[str],
) -> None:
    if left == right or not graph.has_node(left) or not graph.has_node(right):
        return
    previous = graph.get_edge_data(left, right) or {}
    merged = list(dict.fromkeys([*previous.get("shared", []), *shared]))
    if graph.has_edge(left, right):
        graph[left][right]["weight"] = max(float(previous.get("weight", 0.0)), weight)
        graph[left][right]["shared"] = merged[:8]
        # A link inside one partition is a stronger teaching signal than a
        # vague word shared by two unrelated subjects.
        if graph.nodes[left]["domain"] == graph.nodes[right]["domain"]:
            graph[left][right]["weight"] += 0.25
        return
    graph.add_edge(left, right, weight=weight, shared=merged[:8], kind="shared-language")


def build_graph() -> nx.Graph:
    """Build the single shared graph from all subject cards."""
    cards = subject_rag.load_cards()
    graph = nx.Graph(name="vido-concept-graph")

    postings: Dict[str, List[int]] = defaultdict(list)
    alias_postings: Dict[str, List[int]] = defaultdict(list)

    for index, card in enumerate(cards):
        node = f"c:{index}"
        domain = subject_rag.SUBJECT_NAMES.get(card.subject_key, card.subject_key)
        graph.add_node(
            node,
            kind=CONCEPT_KIND,
            index=index,
            card=card,
            domain=card.subject_key,
            partition=domain,
            title=card.title,
        )
        for alias in card.aliases:
            key = _normalize(alias)
            if len(key) >= 3:
                alias_postings[key].append(index)
        seen: Set[str] = set()
        for token in _tokens(" ".join([card.title, *card.aliases, card.summary])):
            if len(token) < 3 or token in QUERY_STOP or token in seen:
                continue
            seen.add(token)
            postings[token].append(index)

    for key, indices in alias_postings.items():
        unique = list(dict.fromkeys(indices))
        if not 2 <= len(unique) <= MAX_ALIAS_POSTINGS:
            continue
        for left, right in combinations(unique, 2):
            _add_edge(graph, f"c:{left}", f"c:{right}", 4.0, [key])

    for token, indices in postings.items():
        unique = list(dict.fromkeys(indices))
        if not 2 <= len(unique) <= MAX_TOKEN_POSTINGS:
            continue
        weight = min(2.4, 0.7 + len(token) * 0.16)
        for left, right in combinations(unique, 2):
            _add_edge(graph, f"c:{left}", f"c:{right}", weight, [token])

    return graph


def graph() -> nx.Graph:
    """Return the cached shared graph, rebuilt if the card store was reloaded."""
    global _graph_cache
    if _graph_cache is None:
        _graph_cache = build_graph()
    return _graph_cache


def partition(domain: str) -> nx.Graph:
    """Return one subdomain view of the shared graph."""
    nodes = [node for node, data in graph().nodes(data=True) if data.get("domain") == domain]
    return graph().subgraph(nodes)


def stats() -> Dict[str, int]:
    """Small introspection helper used by tests and diagnostics."""
    current = graph()
    cross = sum(
        1
        for left, right in current.edges()
        if current.nodes[left]["domain"] != current.nodes[right]["domain"]
    )
    return {
        "cards": sum(1 for _, data in current.nodes(data=True) if data.get("kind") == CONCEPT_KIND),
        "edges": current.number_of_edges(),
        "cross_domain_edges": cross,
        "domains": len({data.get("domain") for _, data in current.nodes(data=True)}),
    }


def _node_for(card: subject_rag.TopicCard) -> Optional[str]:
    for node, data in graph().nodes(data=True):
        if data.get("kind") == CONCEPT_KIND and data.get("card") is card:
            return node
    return None


def _concept_candidates(question: str) -> List[subject_rag.TopicCard]:
    """Rank concept nodes against a question without keyword-penalty routing.

    The subject router deliberately punishes unmatched words so it never
    guesses. Concept questions are different: "how does X relate to Y" contains
    framing words that are not topics. Here framing is ignored, aliases are
    scored by informativeness, and the graph's own neighbours get a small boost
    so connected concepts surface together.
    """
    current = graph()
    text = f" {_normalize(question)} "
    content = [token for token in _tokens(question) if len(token) >= 3 and token not in QUERY_STOP]
    content_set = set(content)
    idf = subject_rag.token_idf()

    scored: Dict[str, Tuple[float, bool]] = {}
    for node, data in current.nodes(data=True):
        if data.get("kind") != CONCEPT_KIND:
            continue
        card = data["card"]
        best_score = 0.0
        best_exact = False
        for alias in card.aliases:
            key = _normalize(alias)
            if len(key) < 3:
                continue
            alias_tokens = key.split()
            informativeness = sum(idf.get(token, 1.0) for token in alias_tokens) / len(alias_tokens)
            if f" {key} " in text:
                score = 45.0 + informativeness * 5.0 + min(12.0, len(alias_tokens) * 4.0)
                exact = True
            else:
                overlap = len(set(alias_tokens) & content_set)
                if not overlap:
                    continue
                score = 18.0 + informativeness * 4.0 + overlap * 5.0
                exact = False
            # A bare generic word on an expanded book card ("book", "theory")
            # is not evidence that the question is about that book.
            if len(alias_tokens) == 1 and idf.get(alias_tokens[0], 0.0) < 3.0 and len(content) >= 4:
                score -= 15.0
            if score > best_score:
                best_score, best_exact = score, exact

        if best_score <= 0:
            continue
        partition = subject_rag.SUBJECT_NAMES.get(card.subject_key, "")
        if content_set & set(_tokens(partition)):
            best_score += 6.0
        if card.subject_key in subject_rag.EXPANDED_SUBJECT_KEYS:
            best_score -= 10.0
        if card.subject_key in {"books", "research_papers"}:
            best_score -= 12.0
        scored[node] = (best_score, best_exact)

    ordered = sorted(scored.items(), key=lambda item: (-item[1][0], not item[1][1], current.nodes[item[0]]["title"]))
    if not ordered:
        return []

    # Graph reasoning: concepts linked to the strongest match are more likely
    # to be the "other" concept the question is asking about.
    leader = ordered[0][0]
    for neighbour, data in current[leader].items():
        if neighbour not in scored or current.nodes[neighbour].get("kind") != CONCEPT_KIND:
            continue
        score, exact = scored[neighbour]
        # Do not let a graph hint reorder concepts the person named explicitly.
        if not exact:
            scored[neighbour] = (score + 6.0, exact)
    ordered = sorted(scored.items(), key=lambda item: (-item[1][0], not item[1][1], current.nodes[item[0]]["title"]))
    return [current.nodes[node]["card"] for node, _ in ordered]


def _neighbours(node: str, limit: int = MAX_NEIGHBOURS) -> List[str]:
    """Strongest linked concepts, preferring the same partition for clarity."""
    current = graph()
    domain = current.nodes[node]["domain"]
    ranked = sorted(
        (
            (neighbour, data.get("weight", 0.0))
            for neighbour, data in current[node].items()
            if current.nodes[neighbour].get("kind") == CONCEPT_KIND
        ),
        key=lambda pair: (
            current.nodes[pair[0]]["domain"] != domain,
            -pair[1],
            current.nodes[pair[0]]["title"],
        ),
    )
    return [candidate for candidate, weight in ranked[:limit] if weight >= 1.5]


def _mode(question: str) -> str:
    if COMPARE_RE.search(question):
        return "compare"
    if RELATE_RE.search(question):
        return "relate"
    if MECHANISM_RE.search(question):
        return "mechanism"
    return "concept"


def _two_concepts(
    candidates: Sequence[subject_rag.TopicCard],
) -> Tuple[subject_rag.TopicCard, subject_rag.TopicCard]:
    if not candidates:
        raise ValueError("at least one concept is required")
    if len(candidates) >= 2:
        return candidates[0], candidates[1]
    # A question can name one concept and imply a second through a strong graph
    # neighbour ("how does entropy relate to information?").
    node = _node_for(candidates[0])
    if node:
        for neighbour in _neighbours(node, limit=8):
            extra = graph().nodes[neighbour]["card"]
            if extra is not candidates[0]:
                return candidates[0], extra
    return candidates[0], candidates[0]


def _shared_words(node_a: str, node_b: str) -> List[str]:
    current = graph()
    data = current.get_edge_data(node_a, node_b) or {}
    return list(data.get("shared", []))[:3]


def _path_line(path: Sequence[str]) -> str:
    current = graph()
    return " → ".join(current.nodes[node]["title"] for node in path)


def _cross_domain_line(card_a: subject_rag.TopicCard, card_b: subject_rag.TopicCard, shared: List[str]) -> str:
    if card_a.subject_key == card_b.subject_key:
        return ""
    domain_a = subject_rag.SUBJECT_NAMES.get(card_a.subject_key, card_a.subject_key)
    domain_b = subject_rag.SUBJECT_NAMES.get(card_b.subject_key, card_b.subject_key)
    bridge = f' through "{shared[0]}"' if shared else ""
    return (
        f"These are filed in different partitions, {domain_a} and {domain_b}, "
        f"but the graph links them{bridge}."
    )


def _evidence_lines(card: subject_rag.TopicCard, remember_label: str = "Remember") -> List[str]:
    """Show every stored field without pretending a plain note is a formula."""
    formula = _clip(card.formula, 260)
    points = _clip(card.points, 260)
    if formula and _looks_like_formula(card.formula):
        lines = [f"Quick form: {formula}"]
        if points:
            lines.append(f"{remember_label}: {points}")
        return lines

    notes: List[str] = []
    for value in (formula, points):
        if value and value.lower() not in {note.lower() for note in notes}:
            notes.append(value)
    return [f"{remember_label}: {'; '.join(notes)}"] if notes else []


def _concept_answer(card: subject_rag.TopicCard) -> str:
    node = _node_for(card)
    parts = [_line(card), *_evidence_lines(card)]

    connected: List[str] = []
    if node:
        domain = graph().nodes[node]["domain"]
        local_links = [
            neighbour for neighbour in _neighbours(node, limit=8)
            if graph().nodes[neighbour]["domain"] == domain
        ][:2]
        for neighbour in local_links:
            other = graph().nodes[neighbour]["card"]
            connected.append(f"{other.title} — {_clip(other.summary, 90)}")
    if connected:
        parts.append("Connected concepts: " + "; ".join(connected) + ".")
    text = "\n".join(parts)
    return text[:MAX_ANSWER_CHARS].rstrip() + ("…" if len(text) > MAX_ANSWER_CHARS else "")


def _mechanism_answer(card: subject_rag.TopicCard) -> str:
    node = _node_for(card)
    parts = [_line(card), *_evidence_lines(card, remember_label="Mechanism")]
    for neighbour in _neighbours(node, limit=1) if node else []:
        other = graph().nodes[neighbour]["card"]
        shared = _shared_words(node, neighbour)
        bridge = f' They share "{shared[0]}" in the graph.' if shared else ""
        parts.append(f"Connected idea: {other.title} — {_clip(other.summary, 140)}.{bridge}")
    text = "\n".join(parts)
    return text[:MAX_ANSWER_CHARS].rstrip() + ("…" if len(text) > MAX_ANSWER_CHARS else "")


def _compare_answer(card_a: subject_rag.TopicCard, card_b: subject_rag.TopicCard) -> str:
    if card_a is card_b:
        return _concept_answer(card_a)
    node_a, node_b = _node_for(card_a), _node_for(card_b)
    parts = [f"Compare {card_a.title} and {card_b.title}.\n"]
    for card in (card_a, card_b):
        parts.append(_line(card))
        parts.extend(_evidence_lines(card))
        parts.append("")

    shared: List[str] = []
    path: Optional[List[str]] = None
    if node_a and node_b:
        if graph().has_edge(node_a, node_b):
            shared = _shared_words(node_a, node_b)
            path = [node_a, node_b]
        else:
            try:
                path = nx.shortest_path(graph(), node_a, node_b)
            except nx.NetworkXNoPath:
                path = None
            if path:
                shared = _shared_words(path[0], path[1])

    difference = (
        f"Key difference: {card_a.title} — {_clip(card_a.points or card_a.summary, 130)}. "
        f"{card_b.title} — {_clip(card_b.points or card_b.summary, 130)}."
    )
    parts.append(difference)
    if path and len(path) > 2:
        parts.append(f"They connect along this chain: {_path_line(path)}.")
    parts.append(_cross_domain_line(card_a, card_b, shared))
    text = "\n".join(part for part in parts if part)
    return text[:MAX_ANSWER_CHARS].rstrip() + ("…" if len(text) > MAX_ANSWER_CHARS else "")


def _relate_answer(card_a: subject_rag.TopicCard, card_b: subject_rag.TopicCard) -> str:
    if card_a is card_b:
        return _mechanism_answer(card_a)
    node_a, node_b = _node_for(card_a), _node_for(card_b)
    parts = [f"How {card_a.title} and {card_b.title} connect.\n"]

    path: Optional[List[str]] = None
    if node_a and node_b:
        try:
            path = nx.shortest_path(graph(), node_a, node_b)
        except nx.NetworkXNoPath:
            path = None
    if path:
        parts.append(f"Path: {_path_line(path)}.")
        parts.append("")
        for node in path[:4]:
            card = graph().nodes[node]["card"]
            parts.append(f"{_line(card)}")
        links: List[str] = []
        for left, right in zip(path, path[1:]):
            shared = _shared_words(left, right)
            if shared:
                links.append(f"{graph().nodes[left]['title']} and {graph().nodes[right]['title']} share \"{shared[0]}\"")
        if links:
            parts.append("The links: " + "; ".join(links) + ".")
    else:
        parts.append(_line(card_a))
        parts.append(_line(card_b))

    parts.append(_cross_domain_line(card_a, card_b, _shared_words(node_a, node_b) if node_a and node_b else []))
    text = "\n".join(part for part in parts if part)
    return text[:MAX_ANSWER_CHARS].rstrip() + ("…" if len(text) > MAX_ANSWER_CHARS else "")


def _first_position(card: subject_rag.TopicCard, question: str) -> int:
    text = f" {_normalize(question)} "
    positions = [
        text.find(f" {key} ")
        for key in (_normalize(alias) for alias in card.aliases)
        if len(key) >= 3 and text.find(f" {key} ") >= 0
    ]
    return min(positions) if positions else 10 ** 6


def answer(question: str) -> Optional[str]:
    """Return a graph-grounded concept answer, or None when evidence is weak."""
    ranked = subject_rag.rank_cards(question)
    if not ranked:
        # The strict router already declined. Only continue when the question's
        # specific words are in the concept vocabulary; otherwise let the
        # encyclopedia handle an unfamiliar qualified topic ("Nash equilibrium").
        content = [token for token in _tokens(question) if len(token) >= 3 and token not in QUERY_STOP]
        if any(token not in _known_tokens() for token in content):
            return None
    candidates: List[subject_rag.TopicCard] = [card for _, _, card in ranked[:MAX_SEEDS]]
    for card in _concept_candidates(question):
        if card not in candidates:
            candidates.append(card)
    candidates = candidates[: MAX_SEEDS + 2]
    if not candidates:
        return None

    card_a, card_b = _two_concepts(candidates)
    mode = _mode(question)
    if mode in {"compare", "relate"}:
        left, right = _first_position(card_a, question), _first_position(card_b, question)
        if left < 10 ** 6 and right < 10 ** 6 and right < left:
            card_a, card_b = card_b, card_a

    if mode == "compare":
        return _compare_answer(card_a, card_b)
    if mode == "relate":
        return _relate_answer(card_a, card_b)
    if mode == "mechanism":
        return _mechanism_answer(card_a)
    return _concept_answer(card_a)


def reset() -> None:
    """Drop cached graph state (used by tests after a card reload)."""
    global _graph_cache, _known_vocabulary
    _graph_cache = None
    _known_vocabulary = None


__all__ = [
    "answer",
    "build_graph",
    "graph",
    "partition",
    "reset",
    "stats",
]
