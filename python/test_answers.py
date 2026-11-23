"""Fast offline checks for the Python answer engine.

Run with: python python/test_answers.py
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

# Match server.py: keep local math.py from shadowing Python's standard library.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.append(HERE)

import document_text  # noqa: E402
import graph_rag  # noqa: E402
import smart_answers  # noqa: E402
import vido_math  # noqa: E402
import wikipedia  # noqa: E402


class MathAnswerTests(unittest.TestCase):
    def test_explains_integration_as_a_concept(self):
        answer = vido_math.handle_query("how is integration happen")
        self.assertIsNotNone(answer)
        self.assertIn("adding up many tiny pieces", answer)
        self.assertIn("constant of integration", answer)
        self.assertNotIn("a*e*h*n*p**2", answer)

    def test_computes_and_explains_a_simple_integral(self):
        answer = vido_math.handle_query("what is integration of x^2")
        self.assertIsNotNone(answer)
        self.assertIn("x**3/3 + C", answer)
        self.assertIn("reverse of differentiation", answer)

    def test_explains_and_computes_derivatives(self):
        concept = vido_math.handle_query("what is differentiation")
        self.assertIsNotNone(concept)
        self.assertIn("instantaneous rate of change", concept)

        result = vido_math.handle_query("what is derivative of x^2")
        self.assertIsNotNone(result)
        self.assertIn("The derivative of x**2 is 2*x", result)


class QuestionRoutingTests(unittest.TestCase):
    def test_strips_conversational_fillers_before_wikipedia_lookup(self):
        with patch.object(
            graph_rag,
            "wikipedia_answer",
            return_value="Nash equilibrium is a stable state of a game.",
        ) as lookup:
            answer = graph_rag.answer("ok so what is nash equilibrium", [], [])

        self.assertIn("Nash equilibrium", answer)
        lookup.assert_called_once_with("what is nash equilibrium")

    def test_topic_phrase_strips_fillers(self):
        self.assertEqual(
            wikipedia.topic_phrase("ok so what is nash equilibrium"),
            "nash equilibrium",
        )
        self.assertEqual(
            wikipedia.topic_phrase("ok what is game theory"),
            "game theory",
        )


class TopicSelectionTests(unittest.TestCase):
    def test_prefers_definitional_article_over_company_with_same_words(self):
        candidates = [
            {
                "title": "Blue Chip Stamps",
                "extract": (
                    "Blue Chip Stamps started as a trading stamps company called Blue Chip Stamp "
                    "Company. It was a competitor of S&H Green Stamps in a loyalty stamp program."
                ),
                "rank": 2,
            },
            {
                "title": "Blue chip (stock market)",
                "extract": (
                    "A blue chip is capital stock of a joint-stock company with a national "
                    "reputation for quality, reliability and the ability to operate profitably."
                ),
                "rank": 7,
            },
        ]
        best = wikipedia._best_extract("what is blue chip", "blue chip", candidates)
        self.assertIsNotNone(best)
        self.assertIn("capital stock", best)

    def test_prefers_exact_article_over_parenthetical_topic(self):
        candidates = [
            {
                "title": "Photosynthesis (board game)",
                "extract": (
                    "Photosynthesis is a strategy board game designed by Hjalmar Hach and "
                    "published by Blue Orange in 2017."
                ),
                "rank": 8,
            },
            {
                "title": "Photosynthesis",
                "extract": (
                    "Photosynthesis is a system of biological processes by which plants convert "
                    "light energy into chemical energy."
                ),
                "rank": 0,
            },
        ]
        best = wikipedia._best_extract("what is photosynthesis", "photosynthesis", candidates)
        self.assertIsNotNone(best)
        self.assertIn("biological processes", best)
        self.assertNotIn("board game", best)


class SmartAnswerTests(unittest.TestCase):
    def test_answers_capital_question_directly(self):
        with patch.object(graph_rag, "wikipedia_answer") as lookup:
            answer = graph_rag.answer("Tell me about capital of india", [], [])
        self.assertIn("New Delhi", answer)
        self.assertIn("since 1911", answer)
        lookup.assert_not_called()

    def test_corrects_common_typo_and_answers_nct_parts(self):
        with patch.object(graph_rag, "wikipedia_answer") as lookup:
            answer = graph_rag.answer("which oart are located in NCT", [], [])
        self.assertIn("11 districts", answer)
        self.assertIn("New Delhi is one district", answer)
        lookup.assert_not_called()

    def test_explains_ncr_and_nct_difference(self):
        answer = smart_answers.answer("so NCR and NCT both are different")
        self.assertIsNotNone(answer)
        self.assertIn("union territory", answer)
        self.assertIn("larger planning region", answer)

    def test_best_city_question_is_not_a_random_article(self):
        general = smart_answers.answer("ok tell me about best city to explore in this world")
        followup = smart_answers.answer("so new york is best city in world")
        self.assertIn("depends on what you enjoy", general)
        self.assertIn("great choice", followup)
        self.assertIn("subjective", followup)

    def test_null_matrix_eigenvalues_are_zero(self):
        singular = vido_math.handle_query("what is eigenvalue of null matrix")
        plural = vido_math.handle_query("ok but what is eigenvalues of null matrix")
        self.assertIn("only eigenvalue", singular)
        self.assertIn("multiplicity n", plural)
        self.assertIn("0", singular)

    def test_nilpotent_eigenvalues_and_skew_determinant(self):
        nilpotent = vido_math.handle_query("what is eigenvalue of nilpotent matrix")
        skew = vido_math.handle_query(
            "tell me determinant of a skew symmetric matrix which is in odd shape"
        )
        typo = vido_math.handle_query(
            "determinant of a skew symmetric matrix which is in dd shape"
        )
        self.assertIn("nilpotent matrix are 0", nilpotent)
        self.assertIn("odd-order skew-symmetric matrix is 0", skew)
        self.assertIn("odd-order", typo)

    def test_combined_math_question_gets_both_answers(self):
        answer = vido_math.handle_query(
            "what is eigenvalue of nilpotent matrix and also tell me determinant of a skew symmetric matrix which is in dd shape"
        )
        self.assertIn("nilpotent matrix are 0", answer)
        self.assertIn("odd-order skew-symmetric matrix is 0", answer)


class DocumentAnswerTests(unittest.TestCase):
    def test_extracts_plain_text_and_rejects_images(self):
        text, truncated = document_text.extract_text(
            "notes.txt", b"Deadline: 2026-12-01\n", "text/plain"
        )
        self.assertIn("2026-12-01", text)
        self.assertFalse(truncated)

        with self.assertRaises(document_text.DocumentError):
            document_text.extract_text("photo.png", b"image", "image/png")

    def test_answers_from_loaded_document_before_wikipedia(self):
        documents = [
            {
                "name": "handbook.txt",
                "text": (
                    "Retry policy. Use three attempts with exponential backoff.\n\n"
                    "Deployment notes. The production freeze starts on 2026-12-01."
                ),
            }
        ]
        with patch.object(graph_rag, "wikipedia_answer") as lookup:
            answer = graph_rag.answer(
                "when does the production freeze start?", [], [], documents=documents
            )

        self.assertIn('From "handbook.txt"', answer)
        self.assertIn("production freeze starts on 2026-12-01", answer)
        lookup.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
