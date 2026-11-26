"""Run the 15 user-supplied advanced questions through the Python answer path."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.append(HERE)

import chat_sweep  # noqa: E402
import evaluation  # noqa: E402
import graph_rag  # noqa: E402
import subject_rag  # noqa: E402
import wikipedia  # noqa: E402


QUESTIONS = [
    "Explain the immune system at an advanced level, including its mechanism, regulation and main points.",
    "Derive the Schur complement at an advanced level and explain its link to block determinants and conditional Gaussian models.",
    "Explain photosynthesis at an advanced level, connecting light reactions, chemiosmosis and carbon fixation.",
    "Explain why nilpotent matrices have all eigenvalues zero and describe their Jordan structure.",
    "Discuss comparative advantage at an advanced level and explain why trade can benefit both parties even when one is absolutely more productive.",
    "Explain plate tectonics at an advanced level, including driving mechanisms, evidence and hazards.",
    "Analyze overfitting at an advanced level using the bias-variance trade-off, model capacity and regularisation.",
    "Explain pH at an advanced level, distinguishing activity from concentration and discussing buffer action.",
    "Discuss federalism at an advanced level, including competence disputes and fiscal incentives.",
    "Explain working memory at an advanced level, including capacity limits, chunking and neural evidence.",
    "Analyze Newton's third law at an advanced level and clarify why action-reaction pairs do not cancel on one body.",
    "Explain compound interest at an advanced level, including effective annual rate and real returns after inflation.",
    "Explain coral reefs at an advanced level, including calcification, bleaching thresholds and feedbacks.",
    "Analyze the Cold War at an advanced level through deterrence, proxy conflicts and arms control.",
    "Explain Python at an advanced level, focusing on the GIL, object overhead and idiomatic performance.",
]


def main() -> int:
    cards = subject_rag.load_cards()
    passed = 0
    with (
        patch.dict(wikipedia.CACHE, {}, clear=True),
        patch.object(wikipedia, "_search", side_effect=evaluation.offline_search),
        patch.object(wikipedia, "_hf_tokens", side_effect=evaluation._regex_tokens),
        patch.object(
            graph_rag,
            "wikipedia_answer",
            side_effect=evaluation.offline_wikipedia_answer,
        ),
    ):
        for index, question in enumerate(QUESTIONS, start=1):
            answer = graph_rag.answer(question, [], [])
            expected = chat_sweep._expected_terms(question, cards)
            failures = []
            if not any(chat_sweep._contains_term(answer, term) for term in expected):
                failures.append("off-topic")
            if chat_sweep._has_badge(answer, expected):
                failures.append("badge")
            if any(phrase in answer.lower() for phrase in chat_sweep.FALLBACK_PHRASES):
                failures.append("fallback")
            if not failures:
                passed += 1
            print(f"\n### {index}. {question}")
            print(f"CHECK: {'PASS' if not failures else 'FAIL: ' + ', '.join(failures)}")
            print(answer)
    print(f"\nTOTAL: {passed}/{len(QUESTIONS)} passed")
    return 0 if passed == len(QUESTIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
