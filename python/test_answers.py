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

import graph_rag  # noqa: E402
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
