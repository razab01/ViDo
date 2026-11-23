"""Curated answers for question shapes that Wikipedia handles too literally.

These are short, deterministic answers for recurring conversational questions
such as "capital of India", "NCR vs NCT", or "best city to explore". They run
before the Wikipedia fallback so chat feels like it understood the question
instead of just searching its last few words.
"""

from __future__ import annotations

import re
from typing import Optional


_FILLER = re.compile(
    r"^(?:ok(?:ay)?|so|and|now|well|yeah|yes|but|also|please)\b[\s,.!]*",
    re.I,
)
_LEAD_IN = re.compile(
    r"^(?:tell me about|what about|what is|what are|what's|who is|describe|explain)\s+",
    re.I,
)
_TYPOS = (
    (re.compile(r"\boart\b", re.I), "part"),
    (re.compile(r"\bthus\b", re.I), "this"),
)

_CAPITALS = {
    "india": ("India", "New Delhi"),
    "bharat": ("India", "New Delhi"),
    "spain": ("Spain", "Madrid"),
    "france": ("France", "Paris"),
    "japan": ("Japan", "Tokyo"),
    "germany": ("Germany", "Berlin"),
    "italy": ("Italy", "Rome"),
    "uk": ("the United Kingdom", "London"),
    "united kingdom": ("the United Kingdom", "London"),
    "usa": ("the United States", "Washington, D.C."),
    "u.s.a.": ("the United States", "Washington, D.C."),
    "united states": ("the United States", "Washington, D.C."),
    "america": ("the United States", "Washington, D.C."),
}

_NCT_PARTS = (
    "The National Capital Territory (NCT) of Delhi has 11 districts: Central Delhi, East Delhi, "
    "New Delhi, North Delhi, North East Delhi, North West Delhi, Shahdara, South Delhi, "
    "South East Delhi, South West Delhi and West Delhi. New Delhi is one district inside the NCT, "
    "not the whole territory."
)

_NCR_VS_NCT = (
    "Yes, they are different. NCT (National Capital Territory of Delhi) is a union territory that "
    "contains New Delhi and is administered by the Delhi government. NCR (National Capital Region) "
    "is a much larger planning region around it, covering the whole NCT plus nearby districts in "
    "Haryana, Uttar Pradesh and Rajasthan."
)

_BEST_CITY = (
    "There isn't one official best city — it depends on what you enjoy. New York is a strong pick "
    "for food, museums and energy; Paris for art and history; Tokyo for food and neighbourhoods; "
    "and Istanbul for history across two continents. Tell me what you like and I can narrow it down."
)

_NEW_YORK_BEST = (
    "New York is a great choice, but 'best' is subjective. It stands out for museums, theatre, food "
    "and neighbourhood variety. Paris, Tokyo and Istanbul are just as strong if you prefer art, "
    "history or a slower pace, so treat New York as one of the top contenders rather than an official winner."
)


def normalize(question: str) -> str:
    """Strip chat lead-ins and fix the small typos people actually type."""
    value = str(question or "").strip()
    while True:
        updated = _FILLER.sub("", value).lstrip()
        if updated == value:
            break
        value = updated
    for pattern, replacement in _TYPOS:
        value = pattern.sub(replacement, value)
    value = re.sub(r"\s+", " ", value).strip(" .!?")
    while True:
        updated = _LEAD_IN.sub("", value, count=1).strip()
        if updated == value:
            break
        value = updated
    return value.strip(" .!?")


def answer(question: str) -> Optional[str]:
    """Return a direct answer for a known conversational shape, else None."""
    text = normalize(question)
    if not text:
        return None

    capital = re.fullmatch(r"(?:the\s+)?capital\s+of\s+([a-z][\w\s.-]+)", text, re.I)
    if capital:
        key = capital.group(1).strip().lower().rstrip(".")
        if key in _CAPITALS:
            country, city = _CAPITALS[key]
            suffix = " It has been India's capital since 1911." if key in ("india", "bharat") else ""
            return f"The capital of {country} is {city}.{suffix}"

    if "nct" in text.lower() and re.search(
        r"\b(?:which|what)\b.*\bparts?\b|\bparts?\b.*\b(?:in|of|located|within)\b",
        text,
        re.I,
    ):
        return _NCT_PARTS

    if "ncr" in text.lower() and "nct" in text.lower() and re.search(
        r"\b(?:different|difference|versus|vs|compare|both|same)\b",
        text,
        re.I,
    ):
        return _NCR_VS_NCT

    if re.search(r"\bbest\b.*\bcity\b|\bcity\b.*\bbest\b", text, re.I) and re.search(
        r"\b(?:explore|visit|travel|world|choose)\b", text, re.I
    ):
        if re.search(r"\b(?:new york|nyc)\b", text, re.I):
            return _NEW_YORK_BEST
        return _BEST_CITY

    return None
