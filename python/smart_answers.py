"""Curated answers for question shapes that Wikipedia handles too literally.

These are short, deterministic answers for recurring conversational questions
such as "capital of India", "NCR vs NCT", or "best city to explore". They run
before the Wikipedia fallback so chat feels like it understood the question
instead of just searching its last few words.
"""

from __future__ import annotations

import difflib
import re
from typing import Optional


_FILLER = re.compile(
    r"^(?:ok(?:ay)?|so|and|now|well|yeah|yes|but|also|please|hi|hello|hey|greetings|"
    r"good\s+(?:morning|afternoon|evening))\b[\s,.!]*",
    re.I,
)
_LEAD_IN = re.compile(
    r"^(?:tell me about|what about|what is|what are|what's|who is|describe|explain)\s+",
    re.I,
)
_TYPOS = (
    (re.compile(r"\boart\b", re.I), "part"),
    (re.compile(r"\bthus\b", re.I), "this"),
    (re.compile(r"\bschur\s+components?\b", re.I), "schur complement"),
    (re.compile(r"\bsecondry\b|\bseconday\b", re.I), "secondary"),
    (re.compile(r"\bteritiary\b|\btertuary\b", re.I), "tertiary"),
    (re.compile(r"\bbredth\b", re.I), "breadth"),
    (re.compile(r"\bbreath\s+first\b", re.I), "breadth first"),
)

_GREETING = re.compile(
    r"^(?:(?:(?:hi|hello|hey|yo|howdy|greetings|namaste)(?:\s+there)?|"
    r"good\s+(?:morning|afternoon|evening))\b"
    r"(?:\s*[,!.]?\s*(?:how\s+are\s+you|how'?s\s+it\s+going|what'?s\s+up|how\s+do\s+you\s+do))?"
    r"|(?:how\s+are\s+you|how'?s\s+it\s+going|what'?s\s+up|how\s+do\s+you\s+do)\b)"
    r"[\s.!?]*$",
    re.I,
)

# Greetings get the same two-layer, deterministic correction as booking: an
# explicit map for common slips, then a conservative fuzzy pass over greeting
# words only. A real request such as "hey book this plan" still fails fullmatch.
_GREETING_TYPOS = (
    (r"\bhii+\b", "hi"),
    (r"\bhelo\b|\bhelloo+\b|\bhelooo\b", "hello"),
    (r"\bhey{2,}a?\b|\bhei\b", "hey"),
    (r"\bgud\b|\bgoood\b|\bgoodd\b", "good"),
    (r"\bmornin\b|\bmorningg\b", "morning"),
    (r"\bafternon\b|\bafternoonn\b", "afternoon"),
    (r"\bevenin\b|\bevenning\b", "evening"),
    (r"\bhow\s+r\s+u\b|\bhw\s+r\s+u\b|\bhow\s+r\s+ya\b", "how are you"),
    (r"\bwat'?s\s+up\b|\bwhas\s+up\b", "what's up"),
)
_GREETING_TYPOS = tuple((re.compile(pattern, re.I), value) for pattern, value in _GREETING_TYPOS)
_GREETING_VOCAB = (
    "hi", "hello", "hey", "yo", "howdy", "greetings", "namaste", "good",
    "morning", "afternoon", "evening", "how", "are", "you", "what's", "up",
    "it's", "going", "do", "there",
)

_SMALL_TALK = re.compile(
    r"^(?:i\s+(?:just\s+)?(?:want|wanna|wanted|would\s+like)(?:\s+to)?\s+(?:chat|talk)|"
    r"(?:let'?s|lets)\s+(?:chat|talk)|can\s+we\s+(?:chat|talk)|"
    r"i\s+feel\s+like\s+(?:chatting|talking)|chat\s+with\s+me)"
    r"[\s.!?]*$",
    re.I,
)

GREETING_ANSWER = (
    "Hi there! Good to see you. How can I help today — a question, a quick chat, or a task "
    "you’d like me to book?"
)

_HOW_ARE_YOU_ANSWER = (
    "I’m doing well, thanks for asking! I’m here for a chat or to help you book a task. "
    "Ask me anything, or attach a plan and say `book this plan`."
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

_FEDERALISM = (
    "Federalism is a system of government where power is divided between a central/national "
    "government and smaller state or regional governments. Both levels have their own "
    "responsibilities — national defence and foreign policy usually sit with the centre, while "
    "education, health and local policing are often regional — and a constitution defines the "
    "boundary. The United States, India, Germany and Brazil are examples."
)

_BRICS = (
    "BRICS is a grouping of major emerging economies that promotes economic and political "
    "cooperation and a more multipolar world. It began as BRIC with Brazil, Russia, India and "
    "China; South Africa joined in 2010, and Egypt, Ethiopia, Iran, the United Arab Emirates "
    "and Indonesia joined later. Membership and partner status can change, so check the latest "
    "official list for a precise current count."
)

_WHO_ARE_YOU = (
    "I’m ViDo’s chat assistant. I answer questions from our conversation, your uploaded data and "
    "documents, and general knowledge. I can also create tasks with dates and flowcharts and show "
    "them inside the chat. Ask me anything, or attach a document and ask about it."
)

_A_STAR = (
    "A* (A-star) is a pathfinding algorithm. At each step it chooses the node with the smallest "
    "f(n) = g(n) + h(n): g(n) is the cost already paid to reach the node, and h(n) is an estimated "
    "cost to the goal. With an admissible and consistent heuristic, A* is complete and finds the "
    "cheapest path. It is widely used in maps, games and robotics."
)

_BREADTH_FIRST = (
    "Breadth-first search (BFS) explores a graph level by level from a starting node, using a queue. "
    "It visits every neighbour at the current depth before moving deeper, so on an unweighted graph "
    "it finds the shortest path by number of edges. Its time complexity is O(V + E)."
)

_ECONOMIC_SECTORS = (
    "Economies are commonly grouped into three sectors. Primary extracts raw materials — farming, "
    "mining, fishing and forestry. Secondary turns those materials into goods — manufacturing, "
    "construction and energy production. Tertiary provides services — transport, retail, healthcare, "
    "education and banking. Some models add a quaternary sector for research, information and technology."
)

_BARTER = (
    "Yes. Barter is still used, usually alongside money: neighbours swap skills or goods, "
    "businesses use trade networks, and communities may rely on it when cash is scarce. It is "
    "less common than money because barter needs a double coincidence of wants — each person "
    "must want what the other offers."
)

_JAVA_LANGUAGE = (
    "Java is a high-level, object-oriented programming language designed to run on a virtual "
    "machine. You write source code, compile it to bytecode, and a JVM on Windows, macOS or Linux "
    "runs that same bytecode. If you meant the Indonesian island, say “Java island” and I’ll answer that instead."
)

_WORA = (
    "WORA stands for Write Once, Run Anywhere. Java source is compiled to platform-neutral "
    "bytecode instead of a Windows or macOS executable. Each platform has its own JVM that reads "
    "and runs that bytecode, so the same .class file works wherever a compatible JVM exists. "
    "Caveat: platform-specific libraries or native code can still break portability."
)

_SCHUR_COMPLEMENT = (
    "For a block matrix M = [[A, B], [C, D]] where A is invertible, the Schur complement of A in M "
    "is S = D - C A⁻¹ B. It comes from block Gaussian elimination and gives det(M) = det(A) · det(S). "
    "If D is invertible instead, the Schur complement of D is A - B D⁻¹ C."
)

_CREATE_DATABASE = (
    "To create a new database, run `CREATE DATABASE my_app_db;` in MySQL or PostgreSQL. "
    "Then select it with `USE my_app_db;` in MySQL or `\\c my_app_db` in PostgreSQL. "
    "In SQLite there is no separate CREATE DATABASE step: connecting to a file such as "
    "`app.db` creates the database. After that, define tables with `CREATE TABLE ...`."
)

_CREATE_COLUMN = (
    "Use `ALTER TABLE`. For example, `ALTER TABLE users ADD COLUMN email VARCHAR(255);` adds an "
    "email column to the users table. Replace the table name, column name and data type as needed. "
    "PostgreSQL supports `ADD COLUMN IF NOT EXISTS`; SQLite supports the basic form, but a NOT NULL "
    "column needs a default value."
)

_SQLALCHEMY = (
    "SQLAlchemy is a Python SQL toolkit and object-relational mapper (ORM). Its Core layer builds "
    "SQL expressions and manages connections; its ORM maps Python classes to database tables so you "
    "query and update with objects. Install it with `pip install sqlalchemy`, create an engine with "
    "`create_engine('sqlite:///app.db')`, then use sessions or Core statements to work with data."
)

_PYTHON_IF = (
    "In Python, `if` is a conditional statement: its indented block runs only when the expression is "
    "truthy. Add `elif` for another branch and `else` for the fallback. For example: "
    "`x = 5`, then `if x > 0:`, `    print('positive')`, `else:`, `    print('not positive')`. "
    "Indentation marks the block, and `if` works with any truthy or falsy value, not just comparisons."
)


def _correct_greeting(text: str) -> str:
    """Fix greeting typos without treating the rest of a message as small talk."""
    value = str(text or "").strip()
    for pattern, replacement in _GREETING_TYPOS:
        value = pattern.sub(replacement, value)

    def fuzzy_token(match: re.Match[str]) -> str:
        token = match.group(0)
        lower = token.lower()
        if lower in _GREETING_VOCAB or len(lower) < 3:
            return token
        closest = difflib.get_close_matches(lower, _GREETING_VOCAB, n=1, cutoff=0.84)
        return closest[0] if closest else token

    return re.sub(r"[A-Za-z']+", fuzzy_token, value)


def greeting_answer(question: str) -> Optional[str]:
    """Return a warm reply for a greeting, not for 'hi what is X'."""
    text = _correct_greeting(question)
    if not _GREETING.fullmatch(text) and not _SMALL_TALK.fullmatch(text):
        return None
    if re.search(r"how\s+are\s+you|how'?s\s+it\s+going|how\s+do\s+you\s+do", text, re.I):
        return _HOW_ARE_YOU_ANSWER
    return GREETING_ANSWER


def correct_typos(question: str) -> str:
    """Fix common spelling slips without stripping the question's lead-in."""
    value = str(question or "")
    for pattern, replacement in _TYPOS:
        value = pattern.sub(replacement, value)
    return re.sub(r"\s+", " ", value).strip()


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
    greeting = greeting_answer(question)
    if greeting:
        return greeting

    text = normalize(question)
    if not text:
        return None

    lower = text.lower()
    if re.fullmatch(
        r"(?:who\s+are\s+you|what\s+are\s+you|introduce\s+yourself|yourself)", lower
    ):
        return _WHO_ARE_YOU

    if re.match(r"^federalism\b", lower):
        return _FEDERALISM

    if re.search(r"\ba\s*\*\s*(?:star\s*)?(?:algorithm|search)\b|\ba\s+star\s+(?:algorithm|search)\b", text, re.I):
        return _A_STAR

    if re.search(r"\bbreadth\s*[- ]?first\s+(?:search|traversal)\b|\bBFS\b", text, re.I):
        return _BREADTH_FIRST

    if re.search(r"\bsectors?\b", text, re.I) and re.search(
        r"\bprimary\b|\bsecondary\b|\btertiary\b", text, re.I
    ):
        return _ECONOMIC_SECTORS

    if re.search(r"\bbrics\b", text, re.I):
        return _BRICS

    if re.search(r"\bWORA\b|\bwrite\s+once[,\s]+run\s+anywhere\b", text, re.I):
        return _WORA

    if re.search(r"\bschur\s+complement\b", text, re.I):
        return _SCHUR_COMPLEMENT

    if re.search(r"\bsql\s*alchemy\b", text, re.I):
        return _SQLALCHEMY

    if (
        re.search(r"\bpython\b", text, re.I)
        and re.search(r"\b(?:if|elif|else)\b", text, re.I)
        and re.search(
            r"\b(?:command|commands|statement|statements|keyword|syntax|condition|"
            r"conditional|use|used|using|work|works)\b",
            text,
            re.I,
        )
    ):
        return _PYTHON_IF

    if re.search(r"\bcolumn\b", text, re.I) and re.search(
        r"\b(?:add|create|make|new)\b", text, re.I
    ):
        return _CREATE_COLUMN

    if re.search(r"\bdatabase\b", text, re.I) and re.search(
        r"\b(?:create|make|start|new)\b", text, re.I
    ):
        return _CREATE_DATABASE

    if re.search(r"\bbarter\b", text, re.I):
        return _BARTER

    if re.search(r"\bjava\b", text, re.I) and not re.search(
        r"\b(?:island|indonesia|sunda|jakarta)\b", text, re.I
    ):
        return _JAVA_LANGUAGE

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
