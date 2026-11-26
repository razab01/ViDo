"""Graph-RAG plan booking helper.

This is deliberately deterministic. Graph RAG owns the conversation; this
helper only understands "book this plan" requests, extracts plan items from the
active document or message, and returns a structured booking state. It does
not add another engine or model, and it never claims a task exists until the
frontend has created it through the normal task callback.
"""

from __future__ import annotations

import copy
import difflib
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple


_MONTH_TOKEN = (
    r"(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|"
    r"aug|august|sep|sept|september|oct|october|nov|november|dec|december)"
)

DATE_TOKEN = (
    r"(?:\d{4}-\d{2}-\d{2}"
    rf"|\d{{1,2}}\s+{_MONTH_TOKEN}\s+\d{{4}}"
    rf"|{_MONTH_TOKEN}\s+\d{{1,2}}(?:,?\s+\d{{4}})?"
    rf"|\d{{1,2}}\s+{_MONTH_TOKEN}"
    r"|day\s+after\s+tomorrow"
    r"|tomorrow|today"
    r"|(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|in\s+\d+\s+days?)"
)

TIME_TOKEN = r"(?:\d{1,2}:\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm))"

_PLAN_NOUN = (
    r"(?:plan|plans|schedule|itinerary|agenda|list|checklist|to-?do|task\s+list)"
)
_QUALIFIED_PLAN = (
    rf"(?:(?:this|the|my|our|these|all|attached|uploaded)\s+{_PLAN_NOUN}"
    rf"|{_PLAN_NOUN}\s+(?:from|in|of)\s+"
    rf"(?:the\s+|this\s+|my\s+|our\s+|attached\s+|uploaded\s+)?"
    rf"(?:document|file|chat|list))"
)

# Two deterministic correction layers keep slightly misspelled requests usable
# without changing the user's actual task title. This is routing cleanup, not a
# new model: an explicit map first, then a conservative fuzzy pass over a small
# routing/date vocabulary.
_EXPLICIT_TYPOS = (
    (r"\b(?:scedul\w*|schedul|schedue|schedual|shedule|shedul|schdule|scheule)\b", "schedule"),
    (r"\b(?:bok|buok|bookk|boook)\b", "book"),
    (r"\b(?:crate|creat|creaet)\b", "create"),
    (r"\b(?:tomm?orow|tomorow|tomorrowr|2moro|2morow)\b", "tomorrow"),
    (r"\b(?:todya|todai|tooday)\b", "today"),
    (r"\b(?:taskt|taskk|taks)\b", "task"),
    (r"\b(?:todot|todoos|toodos)\b", "todo"),
    (r"\b(?:remider|remindr)\b", "reminder"),
    (r"\b(?:mornig|moring)\b", "morning"),
    (r"\b(?:evning|evenin)\b", "evening"),
    (r"\b(?:pls|plz)\b", "please"),
    (r"\b(?:acording|aording)\b", "according"),
    (r"\b(?:documnet|docuemnt|documet)\b", "document"),
)
_EXPLICIT_TYPOS = tuple((re.compile(pattern, re.I), value) for pattern, value in _EXPLICIT_TYPOS)
_INTENT_VOCAB = (
    "book", "schedule", "create", "add", "save", "turn", "put", "do", "setup",
    "task", "tasks", "todo", "todos", "reminder", "reminders", "plan", "plans",
    "agenda", "list", "handbook", "document", "file", "attached", "uploaded",
    "today", "tomorrow", "morning", "evening", "please", "according", "walk",
    "meeting", "review", "payroll",
)

_TASK_NOUN = r"(?:to-?do(?:\s+task)?|tasks?|todos?|reminders?|plans?|agenda|list|items?)"
_SINGLE_TASK_NOUN = r"(?:to-?do(?:\s+task)?|tasks?|todos?|reminders?)"

_DOC_REFERENCE = (
    r"(?:it|this|that|them|these|all|"
    r"(?:the|this|that|attached|uploaded)\s+"
    r"(?:document|file|handbook|plan|list|agenda|attachment))"
)
_PLAN_OBJECT = (
    rf"{_QUALIFIED_PLAN}|task\s+list|(?:these|the|all)\s+tasks|"
    rf"(?:tasks?|plans?|items?)\s+"
    rf"(?:from|in|of|inside|within|based\s+on|according\s+to)\s+{_DOC_REFERENCE}"
)
_BOOKING_RE = re.compile(
    rf"\b(?:book|schedule|create|add|save|put|turn)\b[^.!?\n]*"
    rf"\b(?:{_PLAN_OBJECT}|{_DOC_REFERENCE})\b"
    rf"|\b(?:{_PLAN_OBJECT})\b[^.!?\n]*"
    rf"\b(?:book|schedule|create|add|save)\b"
    rf"|^\s*(?:book|schedule|add|save|create|turn)\s+"
    rf"(?:this|these|it|them|all)\b"
    rf"(?!\s+(?:document|file|question))",
    re.I,
)
_DOCUMENT_REFERENCE_ONLY_RE = re.compile(
    rf"^(?:task|tasks|plan|plans|items?|it|this|that|them|these)?\s*"
    rf"(?:from|in|of|inside|within|based\s+on|according\s+to)\s+"
    rf"{_DOC_REFERENCE}[\s.!]*$",
    re.I,
)

# Polite lead-ins and small-talk openers are removed only for routing. The
# original wording is still used when a title needs to be extracted.
_ROUTING_PREFIX_RE = re.compile(
    r"^\s*(?:(?:hey|hi|hello|yo|howdy|ok|okay|so|well|please|pls|plz|thanks|"
    r"thank\s+you)\b[\s,.:-]*)*"
    r"(?:(?:i\s+)?(?:wanna|want|wanted|would\s+like|need)\s+to\s+)?"
    r"(?:(?:i\s+)?(?:would|will)\s+like\s+to\s+)?",
    re.I,
)
_SINGLE_BOOKING_VERB = r"(?:book|schedule|do|set\s*up)"
_SINGLE_TASK_BOOKING_RE = re.compile(
    rf"^{_SINGLE_BOOKING_VERB}\s+"
    rf"(?:(?:a|an|one|new|my|the|this|that)\s+)?"
    rf"(?:single\s+)?{_SINGLE_TASK_NOUN}\b",
    re.I,
)
# "book a plan for tomorrow ... as ..." describes one new task. Definite forms
# such as "book the plan" still point at an attachment, so only indefinite
# articles are treated as a described plan here.
_SINGLE_PLAN_BOOKING_RE = re.compile(
    rf"^{_SINGLE_BOOKING_VERB}\s+"
    rf"(?:(?:a|an|one|new)\s+)?"
    rf"plans?\b",
    re.I,
)
_BOOKING_COMMAND_RE = re.compile(
    rf"^\s*{_SINGLE_BOOKING_VERB}\s+"
    rf"(?:(?:a|an|one|new|my|the|this|that)\s+)*"
    rf"(?:single\s+)?{_SINGLE_TASK_NOUN}\b[\s,:;-]*",
    re.I,
)
_PLAN_COMMAND_RE = re.compile(
    rf"^\s*{_SINGLE_BOOKING_VERB}\s+"
    rf"(?:(?:a|an|one|new)\s+)*"
    rf"plans?\b[\s,:;-]*",
    re.I,
)
_DOC_REF_REQUEST_RE = re.compile(
    rf"\b(?:from|in|of|inside|within|based\s+on|according\s+to)\s+"
    rf"(?:(?:the|this|that|my|our|attached|uploaded)\s+)?"
    rf"(?:document|file|handbook|attachment|plan|list|it|this|that|them|these)\b",
    re.I,
)

# Short definite requests such as "book the task" or "schedule task" refer to
# the loaded plan. The optional "from it" tail keeps "book todo from it" on the
# document path without swallowing "book todo task for tomorrow".
_DOC_BOOKING_RE = re.compile(
    r"^\s*(?:(?:please\s+)?(?:book|schedule|create|add|save|turn)\s+"
    r"(?:(?:the|this|that|my|our|all|these|attached|uploaded)\s+)?"
    r"(?:to-?do\s+list|tasks?|to-?do|todos?|plans?|agenda|list|handbook|"
    r"it|this|these|them)\b)"
    rf"(?:\s+(?:from|in|of|inside|within|based\s+on|according\s+to)\s+{_DOC_REFERENCE})?"
    r"[\s.!]*$",
    re.I,
)

_HEADER_RE = re.compile(
    r"^(?:(?:my|the|our)?\s*(?:plan|plans|schedule|agenda|tasks?|to-?do|checklist)\s*:?|"
    r"(?:day|week|phase|part)\s+\d+\s*:?)$",
    re.I,
)

_TITLE_HEADER_RE = re.compile(
    r"^(?:(?:my|the|our|project|weekly|monthly|daily|study|work|exam|final|team|personal)\s+)?"
    r"(?:plan|plans|schedule|agenda|to-?do|checklist)\b",
    re.I,
)

_METADATA_CONTINUATION_RE = re.compile(
    r"^(?:due|start|deadline|end|time|at|priority|category|steps?|flowchart)\b",
    re.I,
)

_START_LABEL_RE = re.compile(
    rf"\b(?:start(?:s|ing)?(?:\s+date)?|from|begin(?:ning|s)?(?:\s+date)?)\s*"
    rf"(?:on|at|:|=)?\s*({DATE_TOKEN})",
    re.I,
)

_DUE_LABEL_RE = re.compile(
    rf"\b(?:due(?:\s+date)?|deadline|by|until|end(?:s|ing)?(?:\s+date)?)\s*"
    rf"(?:on|at|:|=)?\s*({DATE_TOKEN})",
    re.I,
)

_DATE_RE = re.compile(DATE_TOKEN, re.I)
_TIME_RE = re.compile(TIME_TOKEN, re.I)
_PRIORITY_RE = re.compile(
    r"\b(?:priority\s*:?\s*(?P<after>high|medium|low)|(?P<before>high|medium|low)\s+priority)\b",
    re.I,
)
_CATEGORY_RE = re.compile(
    r"\b(?:category\s*:?\s*(?P<after>work|home|personal)|"
    r"(?:for\s+)?(?P<before>work|home|personal)\s+(?:task|category))\b",
    re.I,
)
_FLOWCHART_RE = re.compile(
    r"(?:\b(?:flowchart|flow\s*chart|steps?|workflow)\s*:?\s*)?"
    r"(?P<steps>[A-Za-z0-9][^->\n]{1,60}(?:\s*->\s*[A-Za-z0-9][^->\n]{1,60}){1,8})",
    re.I,
)
_TRAILED_CATEGORY_RE = re.compile(r"(?:^|[,;]\s*)(work|home|personal)\s*$", re.I)
_FILLER_RE = re.compile(
    r"^(?:(?:ok|okay|so|now|and|well|please|hey|hi|yes|yeah|yep|sure)\b[\s,.:-]*)*",
    re.I,
)
_LEADING_FOR_RE = re.compile(r"^\s*(?:for|about)\s+", re.I)
# After a date/time is peeled off, phrases like "for tomorrow 4am as meet ..."
# leave "for ... as" before the real title. Strip only leading connectors;
# "to-do" and other genuine titles stay untouched.
_LEADING_CONNECTOR_RE = re.compile(
    r"^\s*(?:for|about|as|at|on|to(?!-))\b[\s,:;-]*",
    re.I,
)
_DANGLING_PREPOSITION_RE = re.compile(
    r"\s+\b(?:for|about|as|on|at|to|from|in|of)\s*$",
    re.I,
)
_DOCUMENT_HINT_RE = re.compile(
    r"\b(?:mentioned|included|listed|stated|given|from|according\s+to|use|based\s+on|"
    r"in|inside|within)\b"
    r"[^.!?\n]*\b(?:document|file|handbook|attachment|it|this|that|above)\b",
    re.I,
)
_SKIP_DATE_RE = re.compile(
    r"^(?:skip|none|no\s+(?:due\s+)?date|leave\s+it)\s*[.!]?$",
    re.I,
)

# Structured meeting-plan documents. These plans are long, table-heavy and
# deliberately do not look like a pasted to-do list, so they get their own
# parser before the generic line-based fallback.
_MEETING_PLAN_HEADER_RE = re.compile(
    r"^\s*MEETING\s+PLAN\s+(\d+)\s*$",
    re.I | re.M,
)
_PAGE_FOOTER_RE = re.compile(
    r"^\s*Meeting\s+Plans\s+Handbook\s*\n\s*Page\s+\d+\s*$",
    re.I | re.M,
)
_AGENDA_TIME_RE = re.compile(r"^\s*\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\s*$")
_AGENDA_STOP_RE = re.compile(
    r"^\s*(?:Process\s+flow|Before,\s+during\s+and\s+after|EXPECTED\s+OUTPUTS|"
    r"Success\s+looks\s+like)\b",
    re.I,
)
_FLOW_INTRO_RE = re.compile(r"^\s*Follow\s+the\s+arrows\b", re.I)
_FLOW_LEGEND = {
    "start / end",
    "process step",
    "decision (yes / no)",
    "flow",
}
_MEETING_LABELS = {
    "purpose",
    "duration",
    "frequency",
    "participants",
    "agenda",
    "process flow",
    "expected outputs",
    "before, during and after",
}

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


@dataclass
class BookingPlan:
    """The graph-RAG reply plus the structured action/state for the UI."""

    answer: str
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    booking: Optional[Dict[str, Any]] = None

    def as_response(self) -> Dict[str, Any]:
        response: Dict[str, Any] = {"answer": self.answer}
        if self.tasks:
            response["actions"] = {"type": "book_plans", "tasks": self.tasks}
        if self.booking:
            response["booking"] = self.booking
        return response


def _today(now: Optional[date]) -> date:
    return now or date.today()


def _parse_date(token: str, now: date) -> Optional[str]:
    value = re.sub(r"\s+", " ", str(token or "").strip().lower())
    value = value.strip(" ,.;:")
    if not value:
        return None

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError:
            return None

    if value == "today":
        return now.isoformat()
    if value == "tomorrow":
        return (now + timedelta(days=1)).isoformat()
    if value == "day after tomorrow":
        return (now + timedelta(days=2)).isoformat()

    match = re.fullmatch(r"in\s+(\d+)\s+days?", value)
    if match:
        return (now + timedelta(days=int(match.group(1)))).isoformat()

    weekday_match = re.fullmatch(r"(?:next\s+)?([a-z]+)", value)
    if weekday_match and weekday_match.group(1) in _WEEKDAYS:
        wanted = _WEEKDAYS[weekday_match.group(1)]
        delta = (wanted - now.weekday()) % 7
        return (now + timedelta(days=delta)).isoformat()

    month_first = re.fullmatch(r"([a-z]+)\s+(\d{1,2})(?:,?\s*(\d{4}))?", value)
    if month_first and month_first.group(1) in _MONTHS:
        year = int(month_first.group(3) or now.year)
        try:
            return date(year, _MONTHS[month_first.group(1)], int(month_first.group(2))).isoformat()
        except ValueError:
            return None

    day_first = re.fullmatch(r"(\d{1,2})\s+([a-z]+)(?:,?\s*(\d{4}))?", value)
    if day_first and day_first.group(2) in _MONTHS:
        year = int(day_first.group(3) or now.year)
        try:
            return date(year, _MONTHS[day_first.group(2)], int(day_first.group(1))).isoformat()
        except ValueError:
            return None

    return None


def _find_dates(text: str, now: date) -> List[Tuple[str, str]]:
    found: List[Tuple[str, str]] = []
    for match in _DATE_RE.finditer(str(text or "")):
        parsed = _parse_date(match.group(0), now)
        if parsed:
            found.append((match.group(0), parsed))
    return found


def _parse_time(token: str) -> Optional[str]:
    value = re.sub(r"\s+", " ", str(token or "").strip().lower())
    match = re.fullmatch(r"(\d{1,2}):(\d{2})\s*(am|pm)?", value)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        meridiem = match.group(3)
        if meridiem:
            if hour == 12:
                hour = 0
            hour += 12 if meridiem == "pm" else 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"
        return None

    match = re.fullmatch(r"(\d{1,2})\s*(am|pm)", value)
    if match:
        hour = int(match.group(1))
        if not 1 <= hour <= 12:
            return None
        if hour == 12:
            hour = 0
        hour += 12 if match.group(2) == "pm" else 0
        return f"{hour:02d}:00"
    return None


def _find_time(text: str) -> Optional[str]:
    for match in _TIME_RE.finditer(str(text or "")):
        parsed = _parse_time(match.group(0))
        if parsed:
            return parsed
    return None


def _normalise_intent(text: str) -> str:
    """Correct booking vocabulary in two conservative, deterministic layers.

    Layer one maps the spelling slips people actually type. Layer two uses a
    small routing/date vocabulary and a high similarity cutoff, so a near miss
    such as ``tommorow`` still routes correctly without rewriting task titles.
    """
    value = str(text or "")
    for pattern, replacement in _EXPLICIT_TYPOS:
        value = pattern.sub(replacement, value)

    def fuzzy_token(match: re.Match[str]) -> str:
        token = match.group(0)
        lower = token.lower()
        if lower in _INTENT_VOCAB or len(lower) < 4:
            return token
        closest = difflib.get_close_matches(lower, _INTENT_VOCAB, n=1, cutoff=0.84)
        return closest[0] if closest else token

    return re.sub(r"\b[A-Za-z]+\b", fuzzy_token, value)


def _strip_routing_prefix(text: str) -> str:
    """Remove greetings and 'I wanted to' only for intent routing."""
    return _ROUTING_PREFIX_RE.sub("", _normalise_intent(text), count=1).strip()


def _block_lines(section: str, label: str, stops: Sequence[str] = ()) -> List[str]:
    match = re.search(rf"(?im)^\s*{re.escape(label)}\s*$", section)
    if not match:
        return []
    tail = section[match.end():]
    end = len(tail)
    for stop in stops:
        stop_match = re.search(rf"(?im)^\s*{re.escape(stop)}\s*$", tail)
        if stop_match:
            end = min(end, stop_match.start())
    return [line.strip() for line in tail[:end].splitlines() if line.strip()]


def _labelled_value(section: str, label: str, stops: Sequence[str] = ()) -> str:
    return re.sub(r"\s+", " ", " ".join(_block_lines(section, label, stops))).strip()


def _meeting_title(section: str) -> str:
    for line in (line.strip() for line in section.splitlines() if line.strip()):
        if re.fullmatch(r"(?:FLOWCHART\s+INCLUDED|AGENDA-ONLY\s+PLAN)", line, re.I):
            continue
        if line.lower() in _MEETING_LABELS:
            break
        return line
    return ""


def _meeting_agenda(section: str) -> List[str]:
    match = re.search(r"(?im)^\s*Agenda\s*$", section)
    if not match:
        return []
    tail = section[match.end():]
    stop = _AGENDA_STOP_RE.search(tail)
    if stop:
        tail = tail[: stop.start()]
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    items: List[str] = []
    index = 0
    while index < len(lines):
        if not _AGENDA_TIME_RE.fullmatch(lines[index]):
            index += 1
            continue
        item = lines[index + 1] if index + 1 < len(lines) else ""
        lead = lines[index + 2] if index + 2 < len(lines) else ""
        if not item or item in {"TIME", "AGENDA ITEM", "LEAD"} or _AGENDA_TIME_RE.fullmatch(item):
            item = ""
        if lead in {"TIME", "AGENDA ITEM", "LEAD"} or _AGENDA_TIME_RE.fullmatch(lead):
            lead = ""
        if item:
            entry = f"{re.sub(r'\s+', ' ', lines[index])} {re.sub(r'\s+', ' ', item)}"
            if lead:
                entry += f" ({re.sub(r'\s+', ' ', lead)})"
            items.append(entry.strip())
        index += 3
    return items


def _meeting_flow(section: str) -> List[str]:
    match = re.search(r"(?im)^\s*Process\s+flow\s*$", section)
    if not match:
        return []
    tail = section[match.end():]
    stop = re.search(r"(?im)^\s*EXPECTED\s+OUTPUTS\b", tail)
    if stop:
        tail = tail[: stop.start()]
    steps: List[str] = []
    for raw in tail.splitlines():
        line = raw.strip()
        if not line or _FLOW_INTRO_RE.match(line):
            continue
        if line.lower() in _FLOW_LEGEND or re.fullmatch(r"yes", line, re.I):
            continue
        continues = bool(steps) and not steps[-1].endswith(("?", ":", "."))
        if continues and (raw[:1].isspace() or line[0].islower()):
            steps[-1] = f"{steps[-1]} {line}"
        else:
            steps.append(line)
    return steps


def _meeting_outputs(section: str) -> List[str]:
    lines = _block_lines(section, "EXPECTED OUTPUTS")
    if not lines:
        match = re.search(r"(?im)^\s*Success\s+looks\s+like:\s*(.+)$", section)
        return [match.group(1).strip()] if match else []
    return [
        cleaned
        for line in lines
        if (cleaned := re.sub(r"^[\s\x7f•*\-\u2022]+", "", line).strip())
    ]


def _meeting_description(
    purpose: str,
    duration: str,
    frequency: str,
    participants: str,
    agenda: Sequence[str],
    flow: Sequence[str],
    outputs: Sequence[str],
) -> str:
    parts: List[str] = []
    for label, value in (
        ("Purpose", purpose),
        ("Duration", duration),
        ("Frequency", frequency),
        ("Participants", participants),
    ):
        if value:
            parts.append(f"{label}: {value}")
    if agenda:
        parts.append("Agenda: " + "; ".join(agenda))
    if flow:
        parts.append("Flow: " + " -> ".join(flow))
    if outputs:
        parts.append("Outputs: " + "; ".join(outputs))
    description = "\n".join(parts)
    return description[:1797].rstrip() + "…" if len(description) > 1800 else description


def _meeting_handbook_tasks(text: str, today: date) -> List[Dict[str, Any]]:
    """Extract the eight structured meeting plans from the handbook PDF."""
    source = str(text or "")
    matches = list(_MEETING_PLAN_HEADER_RE.finditer(source))
    if len(matches) < 2:
        return []

    tasks: List[Dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        section = _PAGE_FOOTER_RE.sub("", source[match.end():end]).strip()
        title = _meeting_title(section)
        if not title:
            continue

        purpose = _labelled_value(section, "PURPOSE", ["DURATION"])
        duration = _labelled_value(section, "DURATION", ["FREQUENCY"])
        frequency = _labelled_value(section, "FREQUENCY", ["PARTICIPANTS"])
        participants = _labelled_value(section, "PARTICIPANTS", ["Agenda"])
        agenda = _meeting_agenda(section)
        flow = _meeting_flow(section)
        outputs = _meeting_outputs(section)
        description = _meeting_description(
            purpose, duration, frequency, participants, agenda, flow, outputs
        )

        tasks.append(
            {
                "title": title,
                "description": description,
                "startDate": "",
                "dueDate": "",
                "dueTime": "",
                "priority": "medium",
                "category": "work",
                "flowchartSteps": flow or agenda,
            }
        )
    return tasks


def _plan_entries(text: str) -> List[str]:
    entries: List[str] = []
    current = ""
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                entries.append(current)
                current = ""
            continue
        line = re.sub(r"^(?:[-*\u2022]|\d+[.)])\s*", "", line).strip()
        if not line or _HEADER_RE.fullmatch(line):
            continue
        # The first "Project Plan"/"Weekly Schedule" line is a document title,
        # not a task to book. Only skip it before any real item has appeared.
        if (
            not current
            and not entries
            and _TITLE_HEADER_RE.match(line)
            and not re.search(r"\d|->|=>|→", line)
        ):
            continue
        if current and _METADATA_CONTINUATION_RE.match(line):
            current = f"{current} {line}"
            continue
        if current:
            entries.append(current)
        current = line
    if current:
        entries.append(current)
    return entries


def _parse_entry(entry: str, now: date) -> Optional[Dict[str, Any]]:
    text = str(entry or "").strip()
    if not text:
        return None

    task: Dict[str, Any] = {
        "title": "",
        "description": "",
        "startDate": "",
        "dueDate": "",
        "dueTime": "",
        "priority": "",
        "category": "",
        "flowchartSteps": [],
    }

    start_match = _START_LABEL_RE.search(text)
    if start_match:
        task["startDate"] = _parse_date(start_match.group(1), now) or ""
        text = text.replace(start_match.group(0), " ")

    due_match = _DUE_LABEL_RE.search(text)
    if due_match:
        task["dueDate"] = _parse_date(due_match.group(1), now) or ""
        text = text.replace(due_match.group(0), " ")

    # An unlabelled date is the deadline; two unlabelled dates stay ambiguous
    # so graph RAG asks instead of silently guessing which one is which.
    remaining_dates = _find_dates(text, now)
    if remaining_dates:
        if len(remaining_dates) == 1:
            if not task["dueDate"]:
                task["dueDate"] = remaining_dates[0][1]
            elif not task["startDate"]:
                task["startDate"] = remaining_dates[0][1]
            text = text.replace(remaining_dates[0][0], " ")
        else:
            task["ambiguousDates"] = [value for _, value in remaining_dates]
            for raw, _ in remaining_dates:
                text = text.replace(raw, " ")

    time_match = _TIME_RE.search(text)
    if time_match:
        task["dueTime"] = _parse_time(time_match.group(0)) or ""
        text = text.replace(time_match.group(0), " ")

    priority_match = _PRIORITY_RE.search(text)
    if priority_match:
        task["priority"] = (priority_match.group("after") or priority_match.group("before") or "").lower()
        text = text.replace(priority_match.group(0), " ")

    category_match = _CATEGORY_RE.search(text)
    if category_match:
        task["category"] = (category_match.group("after") or category_match.group("before") or "").lower()
        text = text.replace(category_match.group(0), " ")

    flowchart_match = _FLOWCHART_RE.search(text)
    if flowchart_match:
        steps = [
            step.strip(" .;:-")
            for step in re.split(r"\s*->\s*", flowchart_match.group("steps"))
            if step.strip(" .;:-")
        ]
        if len(steps) >= 2:
            task["flowchartSteps"] = steps[:8]
            text = text.replace(flowchart_match.group(0), " ")

    trailing_category = _TRAILED_CATEGORY_RE.search(text)
    if trailing_category:
        if not task["category"]:
            task["category"] = trailing_category.group(1).lower()
        text = text[: trailing_category.start()]

    title = re.sub(r"\s+", " ", text).strip(" \t-–—:;,|")
    title = _FILLER_RE.sub("", title).strip(" \t-–—:;,|")
    # "book a task for tomorrow at 4am as meet ..." leaves "for ... as" once
    # the date and time are removed; peel those connectors off the title.
    while True:
        cleaned = _LEADING_CONNECTOR_RE.sub("", title).strip(" \t-–—:;,|")
        if cleaned == title:
            break
        title = cleaned
    while True:
        cleaned = _DANGLING_PREPOSITION_RE.sub("", title).strip(" \t-–—:;,|")
        if cleaned == title:
            break
        title = cleaned
    task["title"] = title
    return task if title or task["dueDate"] else None


def extract_plan_tasks(text: str, now: Optional[date] = None) -> List[Dict[str, Any]]:
    """Extract plan items from a plan document or pasted list."""
    today = _today(now)
    handbook_tasks = _meeting_handbook_tasks(text, today)
    if handbook_tasks:
        return handbook_tasks
    tasks: List[Dict[str, Any]] = []
    for entry in _plan_entries(text):
        task = _parse_entry(entry, today)
        if task:
            tasks.append(task)
    return tasks


def _wants_booking(question: str, has_documents: bool = False) -> bool:
    normalised = _strip_routing_prefix(question)
    if _SINGLE_TASK_BOOKING_RE.match(normalised):
        return True
    if _SINGLE_PLAN_BOOKING_RE.match(normalised):
        return True
    if _BOOKING_RE.search(normalised):
        return True
    return has_documents and bool(_DOC_BOOKING_RE.search(normalised))


def _uses_document_source(normalised: str, has_documents: bool) -> bool:
    """Decide whether this booking turn refers to the attachment or the message."""
    if not has_documents:
        return False
    if _DOC_REF_REQUEST_RE.search(normalised):
        return True
    if _DOC_BOOKING_RE.search(normalised):
        return True
    # "book a task for morning walk tomorrow" describes a new task; it must not
    # be answered from whatever document happens to be attached. The same goes
    # for the indefinite "book a plan for ... as ..." wording.
    if _SINGLE_TASK_BOOKING_RE.match(normalised):
        return False
    if _SINGLE_PLAN_BOOKING_RE.match(normalised):
        return False
    return True


def _document_text(documents: Optional[Sequence[dict]]) -> str:
    if not documents:
        return ""
    return "\n".join(str(document.get("text") or "") for document in documents).strip()


def _clean_title(raw: str) -> str:
    text = re.sub(r"\s+", " ", str(raw or "")).strip()
    text = _FILLER_RE.sub("", text).strip(" \t-–—:;,|")
    return text[:120]


def _next_gap(tasks: List[Dict[str, Any]], today: date) -> Optional[Dict[str, Any]]:
    # Prepare every item before choosing a question. A long handbook has many
    # missing due dates, but that is one group decision, not eight separate
    # interruptions.
    for index, task in enumerate(tasks):
        title = _clean_title(task.get("title"))
        if title:
            task["title"] = title
        else:
            return {
                "taskIndex": index,
                "field": "title",
                "question": "What should I call this task?",
            }

        ambiguous = task.pop("ambiguousDates", None)
        if ambiguous:
            return {
                "taskIndex": index,
                "field": "dueDate",
                "question": (
                    f'I found more than one date for "{title}" — {", ".join(ambiguous)}. '
                    "Which one should I use as the due date?"
                ),
            }

        if not task.get("startDate"):
            task["startDate"] = today.isoformat()

    missing = [index for index, task in enumerate(tasks) if not task.get("dueDate")]
    if len(missing) > 1:
        return {
            "taskIndex": missing[0],
            "taskIndices": missing,
            "group": True,
            "field": "dueDate",
            "question": (
                "I’ve got the details — now I just need one date for all "
                f"{len(missing)} plan items. You can say tomorrow, 2030-12-01, "
                "or `skip` to use today."
            ),
        }
    if missing:
        index = missing[0]
        return {
            "taskIndex": index,
            "field": "dueDate",
            "question": (
                f'I’ve got "{tasks[index]["title"]}" ready. What due date should I use? '
                "You can say tomorrow, 2030-12-01, or `skip` to use today."
            ),
        }

    for index, task in enumerate(tasks):
        if task["dueDate"] < task["startDate"]:
            return {
                "taskIndex": index,
                "field": "dueDate",
                "question": (
                    f'The due date for "{task["title"]}" is earlier than its start date. '
                    "What date should I use instead?"
                ),
            }
    return None


def _advance(
    tasks: List[Dict[str, Any]],
    today: date,
    prefix: str = "",
) -> BookingPlan:
    pending = _next_gap(tasks, today)
    if pending:
        question = pending["question"]
        answer = f"{prefix} {question}".strip() if prefix else question
        return BookingPlan(answer, [], {"tasks": tasks, "pending": pending})

    count = len(tasks)
    answer = (
        f"Got it — I’m booking {count} task now."
        if count == 1
        else f"Got it — I’m booking {count} tasks now."
    )
    if prefix:
        answer = f"{prefix} {answer}"
    return BookingPlan(answer, tasks, None)


def _date_reply_value(text: str, field: str, today: date) -> Tuple[Optional[str], Optional[str]]:
    dates = _find_dates(text, today)
    if len(dates) > 1:
        choices = ", ".join(raw for raw, _ in dates)
        return None, f"I spotted {choices}. Which one should I use?"
    if not dates:
        label = "due date" if field == "dueDate" else "start date"
        return None, f"I didn’t catch a {label}. Try tomorrow or 2026-11-30."
    return dates[0][1], None


def _apply_reply(
    tasks: List[Dict[str, Any]],
    pending: Dict[str, Any],
    reply: str,
    today: date,
) -> Optional[str]:
    """Merge a clarification reply. Returns a follow-up question if still unclear."""
    field = str(pending.get("field") or "")
    text = str(reply or "").strip()

    if field == "title":
        title = _clean_title(text)
        if not title:
            return "What should I call this task?"
        index = int(pending.get("taskIndex") or 0)
        if not 0 <= index < len(tasks):
            return "I lost that task. Please send the details again."
        tasks[index]["title"] = title
        return None

    if field in {"dueDate", "startDate"}:
        indices = pending.get("taskIndices")
        if _SKIP_DATE_RE.fullmatch(text):
            parsed, problem = today.isoformat(), None
        elif _DOCUMENT_HINT_RE.search(text) and not _find_dates(text, today):
            count = len(indices) if isinstance(indices, list) and indices else 1
            scope = f"all {count} plan items" if count > 1 else "this plan item"
            return (
                f"There’s no calendar due date in the details you shared. Give one date for "
                f"{scope}, like tomorrow or 2030-12-01, or say `skip` to use today."
            )
        else:
            parsed, problem = _date_reply_value(text, field, today)
        if problem or parsed is None:
            return problem
        if field == "dueDate" and isinstance(indices, list) and len(indices) > 1:
            if any(not isinstance(index, int) or not 0 <= index < len(tasks) for index in indices):
                return "I lost those tasks. Please send the details again."
            time_value = _find_time(text)
            for index in indices:
                tasks[index]["dueDate"] = parsed
                if time_value:
                    tasks[index]["dueTime"] = time_value
            return None

        index = int(pending.get("taskIndex") or 0)
        if not 0 <= index < len(tasks):
            return "I lost that task. Please send the details again."
        tasks[index][field] = parsed
        if field == "dueDate":
            time_value = _find_time(text)
            if time_value:
                tasks[index]["dueTime"] = time_value
        return None

    if field == "dueTime":
        time_value = _find_time(text)
        if not time_value:
            return "What time should I use? Try 9:30 am or 5 pm."
        index = int(pending.get("taskIndex") or 0)
        if not 0 <= index < len(tasks):
            return "I lost that task. Please send the details again."
        tasks[index]["dueTime"] = time_value
        return None

    return "Could you add that detail again?"


def plan(
    question: str,
    documents: Optional[Sequence[dict]] = None,
    booking: Optional[dict] = None,
    now: Optional[date] = None,
) -> Optional[BookingPlan]:
    """Return a booking plan when graph RAG should own this turn, else None."""
    today = _today(now)

    if booking:
        tasks = copy.deepcopy(list(booking.get("tasks") or []))
        pending = dict(booking.get("pending") or {})
        if not tasks:
            return BookingPlan("Please send the details again so I can book them.")
        follow_up = _apply_reply(tasks, pending, question, today)
        if follow_up:
            return BookingPlan(follow_up, [], {"tasks": tasks, "pending": pending})
        return _advance(tasks, today)

    document_source = _document_text(documents)
    has_document = bool(document_source)
    if not _wants_booking(question, has_document):
        return None

    normalised = _strip_routing_prefix(question)
    no_plan_reply = BookingPlan(
        "I couldn't find a plan list to book. Attach the plan or paste the items here, "
        "and I’ll turn them into tasks."
    )

    # "book a task from it" without an attachment has nothing to read. Stop
    # before the instruction itself becomes the task title.
    if _DOC_REF_REQUEST_RE.search(normalised) and not has_document:
        return no_plan_reply

    if _uses_document_source(normalised, has_document):
        source = document_source
    else:
        command = _BOOKING_COMMAND_RE.match(normalised) or _PLAN_COMMAND_RE.match(normalised)
        if command:
            source = _LEADING_FOR_RE.sub("", normalised[command.end():]).strip()
        else:
            # Pasted lists such as "book this plan: - Draft due tomorrow".
            source = re.sub(
                r"^(?:(?:please\s+)?(?:book|schedule|create|add|save|put|turn)\s+"
                r"(?:this|these|it|them|the|a|an)?\s*"
                r"(?:plan|plans|schedule|list|tasks?|to-?do|todos?)?\s*[:,\-]?\s*)",
                "",
                normalised,
                flags=re.I,
            ).strip()
            if _DOCUMENT_REFERENCE_ONLY_RE.fullmatch(source):
                source = ""
        if not source:
            return BookingPlan(
                "Tell me what you’d like to book and when — like "
                "`book a task for morning walk tomorrow at 4am`."
            )

    tasks = extract_plan_tasks(source, today)
    if not tasks:
        return no_plan_reply
    return _advance(tasks, today)


def continue_plan(
    booking: dict,
    reply: str,
    now: Optional[date] = None,
) -> Optional[BookingPlan]:
    """Continue an in-flight booking. Kept explicit for direct graph-RAG callers."""
    return plan(reply, None, booking, now)
