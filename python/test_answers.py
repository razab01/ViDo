"""Fast offline checks for the Python answer engine.

Run with: python python/test_answers.py
"""

from __future__ import annotations

import math
import os
import re
import sys
import unittest
from datetime import date, timedelta
from unittest.mock import patch

# Match server.py: keep local math.py from shadowing Python's standard library.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.append(HERE)

import arxiv  # noqa: E402
import concept_graph  # noqa: E402
import document_text  # noqa: E402
import evaluation  # noqa: E402
import graph_rag  # noqa: E402
import physics_planners as pp  # noqa: E402
import plan_booking  # noqa: E402
import smart_answers  # noqa: E402
import subject_rag  # noqa: E402
import subject_solvers  # noqa: E402
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

    def test_counts_digit_arrangements_without_repetition(self):
        answer = vido_math.handle_query(
            "how many 5 digit number can be formed by 1,2,3,4,5"
        )
        self.assertIsNotNone(answer)
        self.assertIn("120", answer)
        self.assertIn("5 × 4 × 3 × 2 × 1", answer)
        self.assertIn("3125", answer)

    def test_counts_partial_length_and_repetition_cases(self):
        cases = {
            "how many 3 digit numbers can be formed from 1,2,3,4,5": "60",
            "how many 5 digit numbers can be formed from 1,2,3,4,5 with repetition allowed": "3125",
            "how many 4 digit numbers can be formed from 0,1,2,3 without repetition": "18",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                answer = vido_math.handle_query(question)
                self.assertIsNotNone(answer)
                self.assertIn(expected, answer)

        zero_case = vido_math.handle_query(
            "how many 4 digit numbers can be formed from 0,1,2,3 without repetition"
        )
        self.assertIn("cannot start with 0", zero_case)

    def test_counting_question_beats_dataset_graph_fallback(self):
        rows = [{"Task": "T14"}, {"Task": "T01"}]
        answer = graph_rag.answer(
            "how many 5 digit number can be formed by 1,2,3,4,5",
            rows,
            ["Task"],
        )
        self.assertIn("120", answer)
        self.assertNotIn("closest records", answer)

    def test_counts_even_and_odd_digit_arrangements(self):
        even = vido_math.handle_query(
            "how many 5 digit even number can be formed by 1,2,3,4,5"
        )
        self.assertIsNotNone(even)
        self.assertIn("48", even)
        self.assertIn("2 × 4 × 3 × 2 × 1", even)
        self.assertIn("1250", even)

        odd = vido_math.handle_query(
            "how many 5 digit odd numbers can be formed by 1,2,3,4,5"
        )
        self.assertIn("72", odd)

        zero_case = vido_math.handle_query(
            "how many 4 digit even numbers can be formed from 0,1,2,3 without repetition"
        )
        self.assertIn("10", zero_case)
        self.assertIn("cannot start with 0", zero_case)

        graph_answer = graph_rag.answer(
            "how many 5 digit even number can be formed by 1,2,3,4,5",
            [{"Task": "T14"}],
            ["Task"],
        )
        self.assertIn("48", graph_answer)
        self.assertNotIn("closest records", graph_answer)

    def test_coin_toss_probability_beats_task_prediction(self):
        question = "what is the probability of getting head if we toss coin single time"
        answer = vido_math.handle_query(question)
        self.assertIsNotNone(answer)
        self.assertIn("1/2", answer)
        self.assertIn("50%", answer)
        self.assertIn("fair coin", answer)

        two_tosses = vido_math.handle_query(
            "probability of at least one head in 2 coin tosses"
        )
        self.assertIn("3/4", two_tosses)

        graph_answer = graph_rag.answer(
            question,
            [{"Task": "T14"}],
            ["Task"],
            suggestions=["Clean the kitchen"],
            trained={"best_model": "BiLSTM"},
        )
        self.assertIn("1/2", graph_answer)
        self.assertNotIn("most likely", graph_answer)

    def test_exactly_single_head_uses_binomial_point_probability(self):
        question = (
            "what is the probability of getting a exactly single head "
            "by tossing coin two time"
        )
        answer = graph_rag.answer(question, [], [])
        self.assertIn("1/2", answer)
        self.assertIn("exactly 1 head", answer)
        self.assertNotIn("at least one head", answer)

        exact_one = vido_math.handle_query(
            "what is the probability of getting exactly one head in two tosses"
        )
        two_heads = vido_math.handle_query(
            "what is the probability of getting two heads in two tosses"
        )
        at_most_one = vido_math.handle_query(
            "what is the probability of getting at most one tail in two tosses"
        )
        self.assertIn("exactly 1 head", exact_one)
        self.assertIn("1/4", two_heads)
        self.assertIn("3/4", at_most_one)

    def test_probability_and_statistics_definitions_are_complete(self):
        overview = graph_rag.answer("what is probability and statistics", [], [])
        for term in ("sample space", "expected value", "hypothesis tests", "central limit theorem"):
            self.assertIn(term, overview)

        mode = graph_rag.answer("what is mode", [], [])
        self.assertIn("occurs most often", mode)
        self.assertNotIn("electromagnetic", mode)

        binomial = graph_rag.answer("what is binomial distribution", [], [])
        self.assertIn("C(n,k)", binomial)
        self.assertIn("np(1-p)", binomial)


class ProbabilityStatisticsTests(unittest.TestCase):
    """Broad, Python-first probability/statistics coverage beyond coins."""

    def test_event_algebra_and_bayes_compute_numerically(self):
        union = graph_rag.answer(
            "If P(A)=0.4, P(B)=0.3 and P(A and B)=0.1, what is P(A or B)?", [], []
        )
        self.assertIn("0.6", union)
        self.assertIn("P(A or B)", union)

        posterior = graph_rag.answer(
            "A disease affects 1% of people. A test is 99% sensitive and has a "
            "5% false positive rate. If someone tests positive, what is the "
            "probability they have the disease?",
            [],
            [],
        )
        self.assertIn("0.1667", posterior)
        self.assertIn("Bayes theorem", posterior)

    def test_distributions_and_moments_compute_numerically(self):
        cases = {
            "What is the probability of exactly 2 successes in 5 independent trials with success probability 0.4?": ("0.3456", "exactly 2 successes"),
            "If X follows a Poisson distribution with lambda=3, what is P(X=2)?": ("0.224", "Poisson"),
            "For a geometric distribution with p=0.25, what is the probability the first success occurs on trial 4?": ("0.1055", "geometric"),
            "If waiting time is exponential with rate 2, what is P(X > 1)?": ("0.1353", "exponential"),
            "Heights are normal with mean 100 and standard deviation 15. What is the z-score for 110?": ("0.6667", "z ="),
            "A discrete random variable has values 1,2,3 with probabilities 0.2,0.3,0.5. What is E[X]?": ("2.3", "E[X]"),
            "A discrete random variable has values 1,2,3 with probabilities 0.2,0.3,0.5. What is Var(X)?": ("0.61", "Var(X)"),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                answer = graph_rag.answer(question, [], [])
                for token in expected:
                    self.assertIn(token, answer)

    def test_dice_cards_urns_and_counting_compute_numerically(self):
        cases = {
            "Two fair dice are rolled. What is the probability the sum is 7?": ("1/6", "36"),
            "A card is drawn from a standard deck. What is the probability it is an ace or a king?": ("2/13", "52"),
            "An urn has 5 red and 3 blue balls. Two balls are drawn without replacement. What is the probability both are red?": ("5/14", "Without replacement"),
            "what is probability of getting sum of 3 from six sided die rolled 3 times": ("1/216", "216"),
            "what is probability of getting sum of 3 from six diced die rolled 3 times": ("1/216", "216"),
            "How many permutations are there of 5 distinct objects?": ("120", "5!"),
            "How many ways can you choose 3 objects from 8 distinct objects?": ("56", "C(8,3)"),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                answer = graph_rag.answer(question, [], [])
                for token in expected:
                    self.assertIn(token, answer)

    def test_adversarial_probability_loopholes_are_closed(self):
        cases = {
            "A fair coin is flipped 3 times. Given that at least one head occurred, what is the probability that exactly two heads occurred?": ("3/7", "Conditioning"),
            "Two fair six-sided dice are rolled. Given that the first die shows 4, what is the probability that the sum is 10?": ("1/6", "fixed at 4"),
            "Two fair six-sided dice are rolled. Given that the first die shows 2, what is the probability that the sum is 10?": ("probability is 0", "0/6"),
            "A fair die is rolled twice. What is the probability that the second roll is strictly greater than the first?": ("5/12", "36"),
            "A fair die is rolled 4 times. What is the probability that the sum is at least 20?": ("35/648", "1296"),
            "From a standard 52-card deck, two cards are drawn without replacement. What is the probability that at least one is an ace?": ("33/221", "C(48,2)"),
            "In a batch of 100 items, 5 are defective. If 10 items are sampled without replacement, what is the probability that exactly 2 are defective?": ("0.0702", "hypergeometric"),
            "Suppose P(A) = 0.6, P(B) = 0.5, and P(A ∩ B) = 0.2. Are A and B independent?": ("No", "≠ 0.3"),
            "A random variable X is uniformly distributed on the interval [2, 8]. What is P(3 ≤ X ≤ 5)?": ("1/3", "Uniform[2, 8]"),
            "If X has a binomial distribution with n = 10 and p = 0.3, what is P(X = 3) + P(X = 4)?": ("0.4669", "C(10,3)"),
            "Thirty people are in a room. Assuming 365 equally likely birthdays and no leap days, what is the probability that at least two share a birthday?": ("0.7063", "birthday"),
            "A fair coin is flipped until the first head appears. What is the probability that the number of flips is odd?": ("2/3", "geometric series"),
            "A bag contains 3 red and 2 blue balls. Two balls are drawn with replacement. What is the probability that both are the same colour?": ("13/25", "same colour"),
            "What is the correlation between X and Y if Cov(X,Y) = 6, Var(X) = 4, and Var(Y) = 9?": ("= 1", "Cov(X,Y)"),
            "A regression line has slope 2.5 and passes through (4, 11). What is the intercept?": ("= 1", "2.5"),
            "For two independent events A and B, P(A) = 0.4 and P(B) = 0.7. What is P(A ∪ B)?": ("0.82", "independent"),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                answer = graph_rag.answer(question, [], [])
                for token in expected:
                    self.assertIn(token, answer)
                for badge in ("Python:", "Local:", "Engine:", "Source:", "Wikipedia:", "arXiv:"):
                    self.assertNotIn(badge, answer)

    def test_second_adversarial_probability_round_is_closed(self):
        cases = {
            "Two fair six-sided dice are rolled. What is the probability that the sum is 7 or 11?": ("2/9", "0.2222"),
            "Three fair six-sided dice are rolled. What is the probability that all three show the same face?": ("1/36", "0.0278"),
            "A fair die is rolled 5 times. What is the probability of at least one six?": ("4651/7776", "0.5981"),
            "Two fair six-sided dice are rolled. Given that at least one die shows a 6, what is the probability that the sum is 10?": ("2/11", "0.1818"),
            "A fair coin is flipped 5 times. Given that the first flip is heads, what is the probability of exactly 3 heads?": ("3/8", "0.375"),
            "From a standard 52-card deck, one card is drawn. Given that it is an ace, what is the probability that it is a spade?": ("1/4", "0.25"),
            "In a batch of 100 items, 5 are defective. If 10 items are sampled without replacement, what is the probability that at least 2 are defective?": ("0.0769", "hypergeometric"),
            "If X has a Poisson distribution with mean 3, what is P(X ≤ 2)?": ("0.4232", "Poisson"),
            "If X is exponential with rate 2, what is P(X < 1)?": ("0.8647", "exponential"),
            "Heights are normally distributed with mean 100 and standard deviation 15. What is P(90 < X < 110)?": ("0.495", "z ="),
            "If X ~ Binomial(20, 0.5), what is P(X ≥ 10)?": ("0.5881", "Binomial"),
            "An urn has 4 red and 6 blue balls. Three balls are drawn without replacement. What is the probability of exactly 2 red balls?": ("3/10", "C(4,2)"),
            "For a random variable X, Var(X) = 5. What is Var(2X + 3)?": ("= 20", "a²Var(X)"),
            "A regression line passes through (1, 3) and (3, 7). What is its slope?": ("= 2", "y₂"),
            "A fair coin is flipped until two consecutive heads appear. What is the expected number of flips?": ("6", "expected number"),
            "A random variable has CDF F(x) = 1 - e^(-2x) for x ≥ 0. What is P(1 < X ≤ 2)?": ("0.117", "F(2)"),
            "Which is more likely: getting at least one six in four rolls of a fair die, or getting at least one double six in 24 rolls of two fair dice?": ("0.5177", "0.4914"),
            "A spam filter flags 90% of spam and 5% of legitimate mail. If 10% of mail is spam, what is the probability that a flagged message is actually spam?": ("2/3", "Bayes theorem"),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                answer = graph_rag.answer(question, [], [])
                for token in expected:
                    self.assertIn(token, answer)
                self.assertNotIn("I understand the question", answer)
                for badge in ("Python:", "Local:", "Engine:", "Source:", "Wikipedia:", "arXiv:"):
                    self.assertNotIn(badge, answer)

    def test_rag_plans_probability_questions_and_python_computes_truth(self):
        cases = {
            "Two cards are drawn from a standard 52-card deck without replacement. What is the probability that the first card is an ace and the second card is a spade?": ("1/52", "0.0192"),
            "The first card drawn from a standard 52-card deck is an ace. Without replacement, what is the probability that the second card is a spade?": ("1/4", "0.25"),
            "Two fair six-sided dice are rolled. Given that the sum is 8, what is the probability that the first die shows 5?": ("1/5", "0.2"),
            "A fair coin is flipped 4 times. Given that exactly two heads occurred, what is the probability that the first flip was heads?": ("1/2", "0.5"),
            "Events A and B satisfy P(A and B) = 0.3 and P(A) = 0.5. What is P(B given A)?": ("0.6",),
            "If P(A given B) = 0.6, P(B given A) = 0.4, and P(B) = 0.5, what is P(A)?": ("0.75",),
            "For events A and B, P(A) = 0.6, P(B given A) = 0.5, and P(B given not A) = 0.2. What is P(B)?": ("0.38",),
            "A disease affects 1 in 1000 people. A test is 99% sensitive and has a 5% false positive rate. If a randomly chosen person tests positive, what is the probability they have the disease?": ("0.0194",),
            "X is exponential with rate 2. What is P(X > 5 given X > 3)?": ("0.0183",),
            "Heights are normally distributed with mean 100 and standard deviation 15. What is the probability that a randomly chosen height is greater than 110?": ("0.2525",),
            "A population has mean 50 and standard deviation 10. For a random sample of size 25, approximate the probability that the sample mean exceeds 52.": ("0.1587",),
            "For random variables X and Y, E[X] = 2, E[Y] = 3, and E[XY] = 8. What is Cov(X,Y)?": ("= 2",),
            "For random variables X and Y, Var(X) = 4, Var(Y) = 9, and Cov(X,Y) = 2. What is Var(X + Y)?": ("= 17",),
            "A fair die is rolled repeatedly. What is the probability that a 6 appears before a 5?": ("1/2",),
            "A fair coin is flipped until either three heads or two tails appear. What is the probability that three heads appear first?": ("5/16", "0.3125"),
            "How many four-digit strings can be formed from the digits 1, 2, 3, and 4 if repetition is allowed?": ("256",),
            "A batch has 20 items, of which 7 are defective. If 8 items are sampled without replacement, what is the probability that more than 2 are defective?": ("0.6084",),
            "A sample of 16 has mean 50 and sample standard deviation 8. What is the 95% confidence interval for the population mean?": ("45.74", "54.26"),
            "For a normal distribution with mean 100 and standard deviation 15, what is the probability of being within 10 units of the mean?": ("0.495",),
            "Calls arrive according to a Poisson process with mean 3 per minute. What is the probability of at least 5 calls in a minute?": ("0.1847",),
            "X is uniformly distributed between 2 and 10. What is the probability that X is between 4 and 7?": ("3/8", "0.375"),
            "The waiting time is exponential with rate 2. What is the probability of waiting more than 3 units?": ("0.0025",),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                answer = graph_rag.answer(question, [], [])
                for token in expected:
                    self.assertIn(token, answer)
                self.assertNotIn("I understand the question", answer)
                for badge in ("Python:", "Local:", "Engine:", "Source:", "Wikipedia:", "arXiv:"):
                    self.assertNotIn(badge, answer)

    def test_inference_and_concept_grounding_are_badge_free(self):
        interval = graph_rag.answer(
            "A sample of 100 has mean 50 and known population standard deviation "
            "10. What is the 95% confidence interval for the population mean?",
            [],
            [],
        )
        self.assertIn("48.04", interval)
        self.assertIn("51.96", interval)

        concepts = {
            "What is Bayes theorem?": ("prior", "posterior"),
            "What is a cumulative distribution function?": ("less than or equal", "F(x)"),
            "What is a Type I error?": ("false positive", "null"),
            "Why does the central limit theorem matter?": ("central limit theorem", "normal"),
        }
        for question, expected in concepts.items():
            with self.subTest(question=question):
                answer = graph_rag.answer(question, [], [])
                for token in expected:
                    self.assertIn(token, answer)
                for badge in ("Python:", "Local:", "Engine:", "Source:", "Wikipedia:", "arXiv:"):
                    self.assertNotIn(badge, answer)

    def test_probability_statistics_typos_still_route_to_math(self):
        cases = {
            "What is bayes theorm?": ("Bayes theorem", "posterior"),
            "What is standrad deviation?": ("square root", "variance"),
            "What is probablity distribution?": ("probability", "distribution"),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                answer = graph_rag.answer(question, [], [])
                for token in expected:
                    self.assertIn(token.lower(), answer.lower())

    def test_source_framing_still_answers_the_real_concept_question(self):
        answer = graph_rag.answer(
            "According to Wikipedia, what is a random variable?", [], []
        )
        self.assertIn("random variable", answer.lower())
        self.assertIn("maps outcomes", answer.lower())
        self.assertNotIn("wikipedia", answer.lower())


class PhysicsPlannerTests(unittest.TestCase):
    """RAG understands physics intent; Python owns the numeric result."""

    def _assert_case(self, question: str, expected) -> None:
        answer = graph_rag.answer(question, [], [])
        for token in expected:
            self.assertIn(token, answer)
        self.assertNotIn("I understand the question", answer)
        for badge in ("Python:", "Local:", "Engine:", "Source:", "Wikipedia:", "arXiv:"):
            self.assertNotIn(badge, answer)

    def test_numerical_physics_planners_compute_truth(self):
        cases = {
            "A 10 kg block slides on a horizontal surface with kinetic friction coefficient 0.2. What is the friction force?": (pp._fmt(19.6), "N"),
            "A 4 kg object moving at 3 m/s has what momentum?": (pp._fmt(12), "kg m/s"),
            "A force of 12 N acts for 5 s. What impulse is delivered?": (pp._fmt(60), "N s"),
            "A 2 kg object at 4 m/s sticks to a 3 kg object at rest. Find the final speed.": (pp._fmt(1.6), "m/s"),
            "A 2 kg mass moves at 3 m/s in a circle of radius 2 m. What centripetal force is needed?": (pp._fmt(9), "N"),
            "A projectile is launched at 20 m/s at 30 degrees on level ground. What is the range?": (pp._fmt(20 ** 2 * math.sin(math.radians(60)) / pp.G0), "m"),
            "A projectile is launched at 20 m/s at 30 degrees on level ground. What is the maximum height?": (pp._fmt((20 * math.sin(math.radians(30))) ** 2 / (2 * pp.G0)), "m"),
            "A 2 kg mass on a spring of constant 50 N/m oscillates. What is the period?": (pp._fmt(2 * math.pi * math.sqrt(2 / 50)), "s"),
            "A simple pendulum of length 2 m oscillates. What is its period?": (pp._fmt(2 * math.pi * math.sqrt(2 / pp.G0)), "s"),
            "A wave has frequency 50 Hz and wavelength 2 m. What is its speed?": (pp._fmt(100), "m/s"),
            "Doppler effect: a sound source emits 500 Hz. An observer moves toward it at 20 m/s. The sound speed is 340 m/s. What frequency is heard?": (pp._fmt(500 * 360 / 340), "Hz"),
            "Doppler effect: a source moving toward an observer at 20 m/s emits 500 Hz. The sound speed is 340 m/s. What frequency is heard?": (pp._fmt(500 * 340 / 320), "Hz"),
            "Using Snell's law, light hits a boundary at 30 degrees. The refractive index is 1.5. What is the refracted angle?": (pp._fmt(math.degrees(math.asin(math.sin(math.radians(30)) / 1.5))), "°"),
            "For a lens, the object distance is 0.30 m and the focal length is 0.10 m. Where is the image?": (pp._fmt(0.15), "m"),
            "What is the Coulomb force between charges of 2e-6 C and 3e-6 C separated by 0.5 m?": (pp._fmt(pp.K_E * 6e-12 / 0.25), "N"),
            "What is the electric field at 2 m from a charge of 2e-6 C?": (pp._fmt(pp.K_E * 2e-6 / 4), "N/C"),
            "What is the equivalent resistance of 6 ohm and 3 ohm resistors in parallel?": (pp._fmt(2), "Ω"),
            "A 1e-5 F capacitor is charged to 100 V. How much energy is stored?": (pp._fmt(0.05), "J"),
            "A photoelectric surface has work function 3e-19 J. Light at 1e15 Hz shines on it. What is the maximum kinetic energy?": (pp._fmt(pp.H_PLANCK * 1e15 - 3e-19), "J"),
            "What is the de Broglie wavelength of a particle with momentum 2e-24 kg m/s?": (pp._fmt(pp.H_PLANCK / 2e-24), "m"),
            "A particle is confined to a position uncertainty of 1e-10 m. What is the minimum momentum uncertainty?": (pp._fmt(pp.HBAR / (2e-10)), "kg m/s"),
            "What is the rest energy of a 0.001 kg mass?": (pp._fmt(0.001 * pp.C_LIGHT ** 2), "J"),
            "A sample starts with 100 atoms and has a half-life of 10 s. How many remain after 20 s?": (pp._fmt(25), "atoms"),
            "What is the Hubble recession speed of a galaxy at 100 Mpc if H0 is 70 km/s/Mpc?": (pp._fmt(7000), "km/s"),
            "An elastic collision between identical carts: a 2 kg cart at 4 m/s hits an identical cart at rest. What is the second cart final speed?": (pp._fmt(4), "m/s"),
            "What is the gravitational potential energy between masses of 2e24 kg and 3e24 kg separated by 1e7 m?": (pp._fmt(pp.G * 2e24 * 3e24 / 1e7), "J"),
            "What is the equivalent resistance of 6 ohm and 3 ohm resistors in series?": (pp._fmt(9), "Ω"),
            "A 100-turn coil of area 0.02 m^2 changes field from 0.1 T to 0.5 T in 2 s. Find induced emf.": (pp._fmt(0.4), "V"),
            "Find the entropy change when 2 mol expands from 1 m^3 to 3 m^3.": (pp._fmt(2 * pp.R_GAS * math.log(3)), "J/K"),
            "What is the pressure at a depth of 10 m in fluid of density 1000 kg/m^3?": (pp._fmt(1000 * pp.G0 * 10), "Pa"),
            "A traveller proper time is 1 s at 2e8 m/s. What time is measured?": (pp._fmt(1 / math.sqrt(1 - (2e8 / pp.C_LIGHT) ** 2)), "s"),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self._assert_case(question, expected)

    def test_advanced_numeric_physics_is_not_replaced_by_a_card(self):
        self._assert_case(
            "Explain the friction force on a 10 kg block with kinetic friction "
            "coefficient 0.2 at an advanced level",
            (pp._fmt(19.6), "N"),
        )

    def test_theoretical_physics_branches_are_badge_free(self):
        cases = {
            "What is String theory?": ("strings", "vibration modes"),
            "Explain Quantum Field Theory at an advanced level, including its formula and the main points to remember.": ("fields", "excitations"),
            "What is Computational Physics Methods?": ("numerical methods", "simulations"),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self._assert_case(question, expected)

    def _assert_adversarial_case(self, question, must_any, forbid_any) -> None:
        answer = graph_rag.answer(question, [], [])
        for pattern in forbid_any:
            self.assertIsNone(re.search(pattern, answer, re.I), answer)
        self.assertTrue(any(re.search(pattern, answer, re.I) for pattern in must_any), answer)
        self.assertNotIn("I understand the question", answer)
        self.assertNotIn("could not parse", answer.lower())
        for badge in ("Python:", "Local:", "Engine:", "Source:", "Wikipedia:", "arXiv:"):
            self.assertNotIn(badge, answer)

    def test_hard_physics_subdomain_cases_use_boundary_aware_computation(self):
        cases = [
            (
                "A 10 kg block rests on a 30 degree incline with coefficient of static friction 0.4. What friction force is required to keep the block from sliding?",
                [r"49\b", r"cannot", r"insufficient", r"exceeds"],
                [r"39\.2"],
            ),
            (
                "A 2 kg cart moving at 5 m/s sticks to a 3 kg cart moving at 2 m/s in the same direction. What is their common speed after the collision?",
                [r"3\.2\b"],
                [r"=\s*2(?:\.0+)?\s*m/s\b", r"(?<![\d.])2(?:\.0+)?\s*m/s\."],
            ),
            (
                "An elastic collision occurs between a 3 kg cart moving at 6 m/s and a 1 kg cart at rest. What are the final velocities?",
                [r"3 m/s", r"9 m/s"],
                [],
            ),
            (
                "A projectile is launched at 20 m/s at 30 degrees from a cliff 20 m above level ground. What is its horizontal range?",
                [r"56\.8|56\.9|57\b"],
                [r"35\.3"],
            ),
            (
                "Two moles of an ideal gas expand from 1 m^3 to 3 m^3 while their temperature rises from 300 K to 600 K. With C_V = 3R/2, what is the entropy change?",
                [r"35\.5|35\.6"],
                [],
            ),
            (
                "Two moles of an ideal gas expand isothermally from 1 L to 3 L at 300 K. What is the entropy change?",
                [r"18\.26|18\.27"],
                [],
            ),
            (
                "A 2 kg block with specific heat 900 J/(kg K) cools from 80 K to 20 K. How much heat is removed?",
                [r"108000|1\.08e\+?5|108 kJ"],
                [],
            ),
            (
                "One mole of a monatomic ideal gas expands adiabatically and reversibly from 400 K to 300 K. What work is done by the gas?",
                [r"1247|1\.25e\+?3|1\.247"],
                [],
            ),
            (
                "Three 6 ohm resistors are connected in parallel. What is the equivalent resistance?",
                [r"= 2\b|2 Ω|2 ohm"],
                [r"= 3\b|3 Ω|3 ohm"],
            ),
            (
                "A 100-turn coil of area 0.02 m^2 rotates in a constant 0.5 T magnetic field from a position where its normal is parallel to the field to one where its normal is perpendicular to the field in 2 s. What is the mean induced emf?",
                [r"0\.5\b"],
                [],
            ),
            (
                "A +2e-6 C charge is at (0, 0) and a +3e-6 C charge is at (4 m, 3 m). What is the magnitude of the electrostatic force between them?",
                [r"2\.16e-3|0\.00216"],
                [],
            ),
            (
                "A 1e-5 F capacitor is charged from 0 V to 100 V and then discharged to 40 V. How much energy is released during the discharge?",
                [r"0\.042\b|4\.2e-2"],
                [r"0\.05\b|5e-2"],
            ),
            (
                "Light travels from glass with refractive index 1.5 into air with refractive index 1.0 and strikes the interface at an angle of 60 degrees to the normal. Using Snell's law, what is the refracted angle?",
                [r"total internal reflection", r"no refracted"],
                [r"35\.26", r"arcsin\(sin\(60°\)/1\.5\)"],
            ),
            (
                "A diverging lens with focal length -10 cm forms an image of an object placed 20 cm from the lens. Where is the image?",
                [r"-6\.6|6\.67 cm", r"virtual"],
                [],
            ),
            (
                "A 500 Hz source moves away from a stationary observer at 40 m/s in air where the speed of sound is 340 m/s. What frequency is heard?",
                [r"447\.4|447\b"],
                [r"568|566"],
            ),
            (
                "Light of wavelength 600 nm passes through a diffraction grating with 500 lines per mm. At what angle does the second-order maximum appear?",
                [r"36\.8|36\.9|36\.87"],
                [],
            ),
            (
                "Light of frequency 1e15 Hz shines on a metal whose work function is 1.5 eV. What is the maximum kinetic energy of the emitted electrons in joules?",
                [r"4\.22e-19|4\.2e-19"],
                [],
            ),
            (
                "An electron with kinetic energy 100 eV has a de Broglie wavelength of what?",
                [r"1\.23e-10|1\.2e-10|0\.123 nm"],
                [],
            ),
            (
                "An electron in a one-dimensional infinite well of width 1 nm transitions from n=3 to n=2. What photon wavelength is emitted?",
                [r"6\.6e-7|6\.6e-07|659 nm|660 nm"],
                [],
            ),
            (
                "A spin-1/2 particle is measured to have spin up along z. If spin along x is measured immediately afterwards, what results and probabilities are possible?",
                [r"prepared spin-up along z.*\+ℏ/2 or −ℏ/2.*50% each"],
                [r"Sᵧ", r"along y"],
            ),
            (
                "A spin-1/2 particle prepared spin-up along z is measured along z. What is the probability of measuring spin up?",
                [r"prepared spin-up along z.*\+ℏ/2 with certainty"],
                [r"Sᵧ", r"50% each"],
            ),
            (
                "A rod has a proper length of 10 m and moves at 0.8c relative to an observer. What length does the observer measure?",
                [r"6\b|6\.0\b"],
                [],
            ),
            (
                "A spaceship moves at 0.6c relative to Earth and launches a probe at 0.6c forward relative to the spaceship. What speed does Earth measure for the probe?",
                [r"0\.88|0\.882"],
                [r"1\.2c"],
            ),
            (
                "A galaxy has redshift z = 0.1 and the Hubble constant is 70 km/s/Mpc. Estimate its recession speed and distance.",
                [r"30000", r"429"],
                [],
            ),
            (
                "A clock is held at r = 2R_s from a non-rotating black hole while a distant clock is far away. What is the ratio of the near clock's rate to the distant clock's rate?",
                [r"0\.707|1/√2|sqrt\(0\.5\)"],
                [],
            ),
            (
                "The density of seawater is 1025 kg/m^3. What is the absolute pressure at a depth of 10 m if atmospheric pressure is 101325 Pa?",
                [r"2017|201775|2\.02e5"],
                [r"100450|100500|1\.00e5"],
            ),
            (
                "Water flows through a horizontal pipe that narrows from 4 m/s to 8 m/s. If density is 1000 kg/m^3, what is the pressure drop?",
                [r"24000|2\.4e4|24 kPa"],
                [],
            ),
            (
                "An electron plasma has density 1e18 m^-3 and electron temperature 10 eV. What is the Debye length in metres?",
                [r"2\.3e-5|2\.35e-5|23 µm"],
                [],
            ),
            (
                "For a homogeneous isotropic medium with P-wave speed 6 km/s and S-wave speed 3.5 km/s, what is Poisson's ratio?",
                [r"0\.24|0\.242"],
                [],
            ),
            (
                "For the explicit finite-difference scheme for the one-dimensional heat equation with diffusivity D and grid spacing delta x, what is the stability limit on the time step delta t?",
                [r"Δt\s*≤\s*Δx²/\(2D\)|dt\s*<=?\s*dx\^2/\(2D\)"],
                [],
            ),
            (
                "If a Monte Carlo estimate has standard error proportional to 1/sqrt(N), by what factor must the sample count increase to reduce the standard error by a factor of 10?",
                [r"100\b|100-fold|factor of 100"],
                [],
            ),
            (
                "A chaotic system has a largest Lyapunov exponent of 0.5 s^-1. By roughly what factor does an initial separation grow after 10 s?",
                [r"148\b|e\^5|150\b"],
                [],
            ),
            (
                "At 310 K, for a monovalent cation with outside concentration 140 mM and inside concentration 14 mM, what is the Nernst equilibrium potential in mV?",
                [r"61\.5|61\b|62\b"],
                [],
            ),
        ]
        for question, must_any, forbid_any in cases:
            with self.subTest(question=question):
                self._assert_adversarial_case(question, must_any, forbid_any)

    def test_advanced_applied_physics_keeps_card_depth(self):
        cases = {
            "Explain the stability limit for the explicit finite-difference heat equation in advanced detail.": (
                "advanced explanation",
                "Computational Physics Methods",
            ),
            "Explain the Debye length in advanced detail.": (
                "advanced explanation",
                "Plasma Physics",
            ),
            "At 310 K, for a monovalent cation with outside concentration 140 mM and inside concentration 14 mM, explain the Nernst equilibrium potential in advanced detail.": (
                "advanced explanation",
                "61.5",
            ),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self._assert_case(question, expected)

    def test_physics_concept_answers_do_not_hijack_statistics_topics(self):
        cases = {
            "What is a Bernoulli distribution?": ("success", "failure"),
            "What is a Bernoulli trial?": ("success", "failure"),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                answer = graph_rag.answer(question, [], [])
                for token in expected:
                    self.assertIn(token, answer.lower(), answer)
                self.assertNotIn("horizontal incompressible flow", answer, answer)


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


class ConversationalAnswerTests(unittest.TestCase):
    def test_answers_federalism_despite_greeting(self):
        with patch.object(graph_rag, "wikipedia_answer") as lookup:
            answer = graph_rag.answer("hi what is federalism", [], [])
        self.assertIn("power is divided", answer)
        self.assertIn("regional", answer)
        lookup.assert_not_called()
        self.assertEqual(wikipedia.topic_phrase("hi what is federalism"), "federalism")

    def test_bare_greeting_gets_a_friendly_invitation(self):
        for greeting in ("hi", "hey", "hello there", "good morning"):
            answer = graph_rag.answer(greeting, [], [])
            self.assertIn("Hi there", answer)
            self.assertIn("Good to see you", answer)
            self.assertIn("How can I help today", answer)
            self.assertNotIn("Type a question", answer)
            self.assertNotIn("whenever you’re ready", answer)

    def test_greeting_never_returns_rude_empty_prompt(self):
        for greeting in ("hey", "hey!", " hey ", "hi", "hello", "hii", "helo"):
            with self.subTest(greeting=greeting):
                answer = graph_rag.answer(greeting, [], [])
                self.assertNotIn("Type a question", answer)
                self.assertTrue(answer.strip())

    def test_greetings_with_small_talk_get_a_conversational_reply(self):
        for greeting in ("hi how are you", "how are you", "how's it going", "how do you do"):
            with self.subTest(greeting=greeting):
                answer = graph_rag.answer(greeting, [], [])
                self.assertIn("doing well", answer)
                self.assertIn("chat", answer)
                self.assertNotIn("Type a question", answer)

        casual = graph_rag.answer("what's up", [], [])
        self.assertIn("Good to see you", casual)

    def test_slightly_misspelled_greetings_stay_conversational(self):
        for greeting in ("hii", "helo", "heyyyya", "gud morning", "how r u"):
            with self.subTest(greeting=greeting):
                answer = graph_rag.answer(greeting, [], [])
                self.assertTrue(answer)
                self.assertNotIn("Type a question", answer)
                self.assertNotIn("Meeting Plans Handbook", answer)
        self.assertIn("Good to see you", graph_rag.answer("gud morning", [], []))
        self.assertIn("doing well", graph_rag.answer("how r u", [], []))
        self.assertIn("Good to see you", graph_rag.answer("I just want to chat", [], []))
        self.assertIn("Good to see you", graph_rag.answer("I wanna talk", [], []))

    def test_typos_still_route_to_the_right_topic(self):
        sectors = graph_rag.answer(
            "what is primary and secondry and teritiary sector?", [], []
        )
        self.assertIn("Primary extracts raw materials", sectors)
        self.assertIn("Tertiary provides services", sectors)

        bfs = graph_rag.answer("what is bredth first search", [], [])
        self.assertIn("Breadth-first search", bfs)
        self.assertIn("queue", bfs)
        self.assertNotIn("Maine", bfs)

    def test_a_star_and_bfs_do_not_fall_back_to_generic_algorithm(self):
        a_star = graph_rag.answer("what is A* Algorithm", [], [])
        self.assertIn("f(n) = g(n) + h(n)", a_star)
        self.assertIn("pathfinding", a_star)
        self.assertNotIn("finite sequence of steps", a_star)

        bfs = graph_rag.answer("what is breadth first search", [], [])
        self.assertIn("level by level", bfs)
        self.assertIn("shortest path", bfs)

    def test_answers_brics_without_misstating_founders(self):
        with patch.object(graph_rag, "wikipedia_answer") as lookup:
            answer = graph_rag.answer("tell me about BRICS?", [], [])
        self.assertIn("South Africa joined in 2010", answer)
        self.assertIn("multipolar", answer)
        lookup.assert_not_called()

    def test_answers_identity_question_about_the_assistant(self):
        with patch.object(graph_rag, "wikipedia_answer") as lookup:
            answer = graph_rag.answer("tell me about who are you", [], [])
        self.assertIn("ViDo’s chat assistant", answer)
        self.assertIn("flowcharts", answer)
        lookup.assert_not_called()

    def test_answers_barter_as_a_yes_no_question(self):
        with patch.object(graph_rag, "wikipedia_answer") as lookup:
            answer = graph_rag.answer("is barter exchange still used somewhere??", [], [])
        self.assertTrue(answer.startswith("Yes"))
        self.assertIn("double coincidence of wants", answer)
        lookup.assert_not_called()

    def test_java_defaults_to_the_language_but_keeps_island_available(self):
        with patch.object(graph_rag, "wikipedia_answer") as lookup:
            answer = graph_rag.answer("What is java?", [], [])
        self.assertIn("programming language", answer)
        self.assertIn("JVM", answer)
        lookup.assert_not_called()
        self.assertIsNone(smart_answers.answer("What is java island?"))

    def test_explains_wora_without_falling_into_dataset_answers(self):
        with patch.object(graph_rag, "wikipedia_answer") as lookup:
            answer = graph_rag.answer("Ok why it is (WORA)", [], [])
        self.assertIn("Write Once, Run Anywhere", answer)
        self.assertIn("bytecode", answer)
        self.assertIn("JVM", answer)
        lookup.assert_not_called()

    def test_explains_schur_complement_and_its_formula(self):
        with patch.object(graph_rag, "wikipedia_answer") as lookup:
            explanation = graph_rag.answer("tell me about schur component.", [], [])
            formula = graph_rag.answer("ok what is formula for schur component", [], [])
        self.assertIn("Schur complement", explanation)
        self.assertIn("D - C A⁻¹ B", explanation)
        self.assertIn("D - C A⁻¹ B", formula)
        self.assertIn("det(M) = det(A) · det(S)", formula)
        lookup.assert_not_called()

    def test_answers_database_questions_directly_without_wikipedia(self):
        with patch.object(graph_rag, "wikipedia_answer") as lookup:
            create = graph_rag.answer("how to create a new database", [], [])
            column = graph_rag.answer("how to create a new column in database", [], [])
            sqlalchemy = graph_rag.answer("what is sqlalchemy", [], [])
        self.assertIn("CREATE DATABASE", create)
        self.assertIn("SQLite", create)
        self.assertIn("ALTER TABLE", column)
        self.assertIn("email VARCHAR(255)", column)
        self.assertIn("object-relational mapper", sqlalchemy)
        self.assertIn("create_engine", sqlalchemy)
        lookup.assert_not_called()

    def test_procedural_questions_are_not_used_as_wikipedia_titles(self):
        with patch.object(wikipedia, "_search") as lookup:
            self.assertIsNone(wikipedia.answer("how to create a new database"))
        lookup.assert_not_called()

    def test_explains_python_if_statement_directly_without_wikipedia(self):
        with patch.object(graph_rag, "wikipedia_answer") as lookup:
            answer = graph_rag.answer("what is if command in python", [], [])
        self.assertIn("conditional statement", answer)
        self.assertIn("elif", answer)
        self.assertIn("print('positive')", answer)
        lookup.assert_not_called()


def _sample_handbook() -> str:
    """A compact stand-in for the attached meeting-plans handbook."""
    plans = [
        ("Project Kickoff Meeting", "Align everyone on goals, scope, roles and timeline."),
        ("Decision-Making Meeting", "Choose between clear options using agreed criteria."),
        ("Incident / Problem-Solving Meeting", "Find the root cause and agree on fixes."),
        ("Sprint Planning", "Decide what the team can deliver in the next sprint."),
        ("Daily Stand-up", "Keep the team in sync and surface blockers early."),
        ("Weekly Team Sync", "Review weekly progress and agree on priorities."),
        ("One-on-One Meeting", "Give and receive feedback, and support growth."),
        ("Brainstorming Session", "Generate many ideas, then narrow to a shortlist."),
    ]
    blocks: List[str] = []
    for number, (title, purpose) in enumerate(plans, 1):
        flow = ""
        outputs = " Written decision with owners"
        if number == 1:
            flow = (
                "Process flow\n"
                "Follow the arrows from start to finish.\n"
                "Start / End\n"
                "Process step\n"
                "Decision (Yes / No)\n"
                "Flow\n"
                "Project approved\n"
                "Define goals and\n"
                "  scope\n"
                "Team aligned?\n"
                "No: Clarify scope, then revisit\n"
                "Kickoff complete\n"
                "Yes\n"
            )
            outputs = " Signed-off project charter\n Shared timeline"
        blocks.append(
            "\n".join(
                [
                    "Meeting Plans Handbook",
                    f"Page {number + 2}",
                    f"MEETING PLAN {number:02d}",
                    title,
                    " FLOWCHART INCLUDED" if number == 1 else " AGENDA-ONLY PLAN",
                    "PURPOSE",
                    purpose,
                    "DURATION",
                    "60 minutes" if number in {1, 3, 8} else "45 minutes",
                    "FREQUENCY",
                    "Once, at project start" if number == 1 else "As needed",
                    "PARTICIPANTS",
                    "Project sponsor, manager and team leads.",
                    "Agenda",
                    " TIME",
                    " AGENDA ITEM",
                    " LEAD",
                    " 0:00 - 0:05",
                    " Review the plan",
                    " Facilitator",
                    flow.rstrip(),
                    "EXPECTED OUTPUTS",
                    outputs,
                    "",
                ]
            )
        )
    return "\n".join(blocks)


class PlanBookingTests(unittest.TestCase):
    def test_document_plan_is_booked_as_canonical_task_actions(self):
        documents = [
            {
                "name": "plan.txt",
                "text": (
                    "Project Plan\n"
                    "- Draft spec due 2030-12-01 priority high category work\n"
                    "- Review design due December 2, 2030 priority medium category home"
                ),
            }
        ]

        result = graph_rag.answer_with_booking(
            "book this plan", [], [], documents=documents
        )

        self.assertNotIn("booking", result)
        actions = result["actions"]
        self.assertEqual(actions["type"], "book_plans")
        self.assertEqual(len(actions["tasks"]), 2)
        first, second = actions["tasks"]
        self.assertEqual(first["title"], "Draft spec")
        self.assertEqual(first["dueDate"], "2030-12-01")
        self.assertEqual(first["priority"], "high")
        self.assertEqual(first["category"], "work")
        self.assertEqual(second["title"], "Review design")
        self.assertEqual(second["dueDate"], "2030-12-02")
        self.assertEqual(second["category"], "home")
        for badge in ("Python", "Local", "Wikipedia", "arXiv", "source:"):
            self.assertNotIn(badge, result["answer"])

    def test_meeting_handbook_sections_extract_eight_detailed_plans(self):
        tasks = plan_booking.extract_plan_tasks(
            _sample_handbook(), now=date(2030, 1, 1)
        )

        self.assertEqual(len(tasks), 8)
        self.assertEqual(tasks[0]["title"], "Project Kickoff Meeting")
        self.assertIn("Align everyone on goals", tasks[0]["description"])
        self.assertIn("0:00 - 0:05 Review the plan", tasks[0]["description"])
        self.assertIn("Define goals and scope", tasks[0]["flowchartSteps"])
        self.assertIn("No: Clarify scope, then revisit", tasks[0]["flowchartSteps"])
        self.assertIn("Signed-off project charter", tasks[0]["description"])
        self.assertEqual(tasks[-1]["title"], "Brainstorming Session")
        for task in tasks:
            self.assertEqual(task["priority"], "medium")
            self.assertEqual(task["category"], "work")
            self.assertTrue(task["flowchartSteps"])

    def test_typoed_document_reference_asks_once_then_books_every_plan(self):
        documents = [
            {
                "name": "Meeting_Plans_Handbook.pdf",
                "text": _sample_handbook(),
            }
        ]

        clarification = graph_rag.answer_with_booking(
            "scedule task according to it", [], [], documents=documents
        )
        self.assertNotIn("actions", clarification)
        self.assertEqual(clarification["booking"]["pending"]["field"], "dueDate")
        self.assertTrue(clarification["booking"]["pending"]["group"])
        self.assertEqual(len(clarification["booking"]["pending"]["taskIndices"]), 8)
        self.assertIn("all 8 plan items", clarification["answer"])

        completed = graph_rag.answer_with_booking(
            "2030-12-01 at 09:30", [], [], booking=clarification["booking"]
        )
        self.assertNotIn("booking", completed)
        tasks = completed["actions"]["tasks"]
        self.assertEqual(completed["actions"]["type"], "book_plans")
        self.assertEqual(len(tasks), 8)
        self.assertTrue(all(task["dueDate"] == "2030-12-01" for task in tasks))
        self.assertTrue(all(task["dueTime"] == "09:30" for task in tasks))
        self.assertEqual(tasks[0]["title"], "Project Kickoff Meeting")
        self.assertIn("Purpose:", tasks[0]["description"])
        for badge in ("Python", "Local", "Wikipedia", "arXiv", "source:"):
            self.assertNotIn(badge, completed["answer"])

    def test_short_document_booking_phrases_use_handbook_plans(self):
        documents = [
            {
                "name": "Meeting_Plans_Handbook.pdf",
                "text": _sample_handbook(),
            }
        ]

        for phrase in ("book the task", "schedule task", "shedule the todo list"):
            with self.subTest(phrase=phrase):
                result = graph_rag.answer_with_booking(
                    phrase, [], [], documents=documents
                )
                self.assertNotIn("actions", result)
                booking = result["booking"]
                self.assertEqual(booking["pending"]["field"], "dueDate")
                self.assertEqual(len(booking["pending"]["taskIndices"]), 8)
                self.assertEqual(booking["tasks"][0]["title"], "Project Kickoff Meeting")

    def test_bare_todo_booking_asks_for_a_name_without_reading_the_document(self):
        documents = [
            {
                "name": "Meeting_Plans_Handbook.pdf",
                "text": _sample_handbook(),
            }
        ]

        result = graph_rag.answer_with_booking(
            "hey wanted to book todo task for tomorrow", [], [], documents=documents
        )

        self.assertNotIn("actions", result)
        booking = result["booking"]
        self.assertEqual(len(booking["tasks"]), 1)
        self.assertTrue(booking["tasks"][0]["dueDate"])
        self.assertFalse(booking["tasks"][0]["title"])
        self.assertEqual(booking["pending"]["field"], "title")
        self.assertEqual(result["answer"], "What should I call this task?")
        for badge in ("Python", "Local", "Wikipedia", "arXiv", "source:", "Meeting Plans Handbook"):
            self.assertNotIn(badge, result["answer"])

    def test_single_described_task_is_booked_from_the_message_not_the_document(self):
        documents = [
            {
                "name": "Meeting_Plans_Handbook.pdf",
                "text": _sample_handbook(),
            }
        ]

        result = graph_rag.answer_with_booking(
            "hey book a task for morning walk for tomorrow at 4am",
            [],
            [],
            documents=documents,
        )

        self.assertEqual(result["actions"]["type"], "book_plans")
        self.assertEqual(len(result["actions"]["tasks"]), 1)
        task = result["actions"]["tasks"][0]
        self.assertEqual(task["title"], "morning walk")
        self.assertEqual(task["dueDate"], (date.today() + timedelta(days=1)).isoformat())
        self.assertEqual(task["dueTime"], "04:00")
        self.assertNotIn("booking", result)
        for badge in ("Python", "Local", "Wikipedia", "arXiv", "source:", "Meeting Plans Handbook"):
            self.assertNotIn(badge, result["answer"])

    def test_book_a_plan_without_details_asks_for_the_plan(self):
        result = graph_rag.answer_with_booking("book a plan", [], [])
        self.assertNotIn("actions", result)
        self.assertNotIn("booking", result)
        self.assertIn("Tell me what you’d like to book", result["answer"])
        self.assertNotIn("closest records", result["answer"])

    def test_book_a_plan_with_details_books_only_the_described_task(self):
        result = graph_rag.answer_with_booking(
            "book a plan for tomorrow 4am as meet with US investors", [], []
        )
        self.assertEqual(result["actions"]["type"], "book_plans")
        self.assertEqual(len(result["actions"]["tasks"]), 1)
        task = result["actions"]["tasks"][0]
        self.assertEqual(task["title"], "meet with US investors")
        self.assertEqual(task["dueDate"], (date.today() + timedelta(days=1)).isoformat())
        self.assertEqual(task["dueTime"], "04:00")
        self.assertNotIn("booking", result)
        for badge in ("Python", "Local", "Wikipedia", "arXiv", "source:", "closest records"):
            self.assertNotIn(badge, result["answer"])

    def test_bare_time_prompt_asks_for_a_title_instead_of_using_at(self):
        result = graph_rag.answer_with_booking("book a task for tomorrow at 9am", [], [])
        self.assertIn("booking", result)
        self.assertEqual(result["booking"]["pending"]["field"], "title")

    def test_do_task_phrase_is_a_booking_request(self):
        result = graph_rag.answer_with_booking("hey do task tomorrow", [], [])
        self.assertNotIn("actions", result)
        self.assertEqual(result["booking"]["pending"]["field"], "title")
        self.assertTrue(result["booking"]["tasks"][0]["dueDate"])

    def test_document_hint_gets_a_specific_question_and_skip_books(self):
        documents = [
            {
                "name": "Meeting_Plans_Handbook.pdf",
                "text": _sample_handbook(),
            }
        ]
        initial = graph_rag.answer_with_booking(
            "schedule task", [], [], documents=documents
        )

        for hint_text in ("mentioned in the document", "it's in document"):
            with self.subTest(hint_text=hint_text):
                hint = graph_rag.answer_with_booking(
                    hint_text, [], [], booking=initial["booking"]
                )
                self.assertIn("no calendar due date", hint["answer"])
                self.assertIn("all 8 plan items", hint["answer"])
                self.assertIn("booking", hint)

        unclear = graph_rag.answer_with_booking(
            "yes", [], [], booking=hint["booking"]
        )
        self.assertNotIn("actions", unclear)
        self.assertIn("booking", unclear)

        completed = graph_rag.answer_with_booking(
            "skip", [], [], booking=hint["booking"]
        )
        self.assertNotIn("booking", completed)
        tasks = completed["actions"]["tasks"]
        self.assertEqual(len(tasks), 8)
        self.assertTrue(all(task["dueDate"] for task in tasks))

    def test_document_reference_without_an_attachment_does_not_invent_a_task(self):
        result = graph_rag.answer_with_booking(
            "scedule task according to it", [], [], documents=[]
        )
        self.assertIn("couldn't find a plan list", result["answer"])
        self.assertNotIn("actions", result)
        self.assertNotIn("booking", result)

    def test_missing_due_date_is_asked_then_booking_resumes(self):
        documents = [
            {
                "name": "plan.txt",
                "text": (
                    "- Draft spec priority high\n"
                    "- Review design due 2030-12-02 category work"
                ),
            }
        ]

        clarification = graph_rag.answer_with_booking(
            "book this plan", [], [], documents=documents
        )
        self.assertNotIn("actions", clarification)
        booking = clarification["booking"]
        self.assertEqual(booking["pending"]["field"], "dueDate")
        self.assertEqual(booking["pending"]["taskIndex"], 0)
        self.assertIn("Draft spec", clarification["answer"])

        completed = graph_rag.answer_with_booking(
            "2030-12-01", [], [], booking=booking
        )
        self.assertNotIn("booking", completed)
        self.assertEqual(completed["actions"]["type"], "book_plans")
        self.assertEqual(completed["actions"]["tasks"][0]["dueDate"], "2030-12-01")
        self.assertEqual(completed["actions"]["tasks"][1]["dueDate"], "2030-12-02")

    def test_ambiguous_date_is_questioned_instead_of_guessed(self):
        documents = [
            {
                "name": "plan.txt",
                "text": "- Draft spec on 2030-12-01 and 2030-12-05",
            }
        ]

        result = graph_rag.answer_with_booking(
            "book this plan", [], [], documents=documents
        )
        self.assertNotIn("actions", result)
        self.assertEqual(result["booking"]["pending"]["field"], "dueDate")
        self.assertIn("more than one date", result["answer"])

    def test_ordinary_question_and_single_task_creation_are_not_booked(self):
        with patch.object(graph_rag, "answer", return_value="ordinary answer") as mocked:
            result = graph_rag.answer_with_booking(
                "create a task to review payroll tomorrow", [], []
            )
        self.assertEqual(result, {"answer": "ordinary answer"})
        mocked.assert_called_once()

    def test_server_answer_endpoint_returns_booking_actions(self):
        import server  # local import keeps the main suite start-up light

        booking = {
            "tasks": [{"title": "Draft spec", "dueDate": ""}],
            "pending": {
                "taskIndex": 0,
                "field": "dueDate",
                "question": "What due date should I use?",
            },
        }
        expected = {
            "answer": "Got it — I’m booking 1 task now.",
            "actions": {
                "type": "book_plans",
                "tasks": [{"title": "Draft spec", "dueDate": "2030-12-01"}],
            },
        }
        body = server.AnswerBody(
            question="book this plan",
            documents=[{"name": "plan.txt", "text": "- Draft spec"}],
            booking=booking,
        )
        with patch.object(server, "answer_with_booking", return_value=expected) as mocked:
            response = server.answer(body)

        self.assertEqual(response, expected)
        arguments = mocked.call_args.args
        self.assertEqual(arguments[0], "book this plan")
        self.assertEqual(arguments[6], body.documents)
        self.assertEqual(arguments[7], booking)


class SubjectRagTests(unittest.TestCase):
    CORE_SUBJECTS = {
        "math", "physics", "chemistry", "biology", "economics",
        "computer_science", "history", "geography", "politics_law", "psychology",
    }
    EXPANDED_SUBJECTS = {
        "books", "research_papers", "physics_branches", "math_branches",
        "statistics_branches", "science_branches", "computing_branches",
        "economics_branches", "humanities_branches", "engineering_branches",
        "health_branches", "probability_statistics",
    }

    def test_store_keeps_core_cards_and_adds_expanded_catalogs(self):
        cards = subject_rag.reload_cards()
        by_subject: dict = {}
        for card in cards:
            by_subject.setdefault(card.subject_key, []).append(card)

        self.assertEqual(
            set(by_subject),
            self.CORE_SUBJECTS | self.EXPANDED_SUBJECTS,
        )
        for subject_key in self.CORE_SUBJECTS:
            with self.subTest(subject=subject_key):
                self.assertEqual(len(by_subject[subject_key]), 50)
        for subject_key in self.EXPANDED_SUBJECTS:
            with self.subTest(subject=subject_key):
                self.assertGreater(len(by_subject[subject_key]), 0)
        for card in cards:
            with self.subTest(card=card.title):
                self.assertTrue(card.aliases and card.summary and card.formula and card.points)

    def test_classifies_expanded_book_and_paper_cards(self):
        cases = {
            "What is Quantum Field Theory?": "Quantum Field Theory",
            "Tell me about Attention Is All You Need": "Attention Is All You Need",
            "What is Classical Dynamics of Particles and Systems?": (
                "Classical Dynamics of Particles and Systems"
            ),
        }
        for question, expected_topic in cases.items():
            with self.subTest(question=question):
                result = subject_rag.classify(question)
                self.assertIsNotNone(result)
                self.assertEqual(result.topic, expected_topic)

    def test_paper_card_answers_never_show_source_names(self):
        cases = {
            "What is Adam A Method for Stochastic Optimization?": "1412.6980",
            "What is arXiv Paper Search?": "preprint archive",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                with patch.object(graph_rag, "wikipedia_answer") as lookup:
                    answer = graph_rag.answer(question, [], [])
                self.assertIn(expected, answer)
                lowered = answer.lower()
                self.assertNotIn("arxiv", lowered)
                self.assertNotIn("wikipedia", lowered)
                lookup.assert_not_called()

    def test_classifies_representative_questions_from_all_ten_subjects(self):
        cases = {
            "What is modular arithmetic?": ("Mathematics", "Modular arithmetic"),
            "Explain the Big Bang theory.": ("Physics", "Big Bang theory"),
            "What is a catalyst?": ("Chemistry", "Catalyst"),
            "What is DNA?": ("Biology", "DNA"),
            "What is opportunity cost?": ("Economics", "Opportunity cost"),
            "Explain a stack data structure.": ("Computer Science", "Stack"),
            "What was the Silk Road?": ("History", "Silk Road"),
            "Explain plate tectonics.": ("Geography", "Plate tectonics"),
            "What is judicial review?": ("Politics and Law", "Judicial review"),
            "What is classical conditioning?": ("Psychology", "Classical conditioning"),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                result = subject_rag.classify(question)
                self.assertIsNotNone(result)
                self.assertEqual((result.subject, result.topic), expected)

    def test_disambiguates_qualified_topics(self):
        self.assertEqual(subject_rag.classify("What is hormone?").topic, "Hormone")
        self.assertEqual(subject_rag.classify("What is depression?").subject, "Psychology")
        self.assertEqual(subject_rag.classify("What is depression economics?").subject, "Economics")

    def test_advanced_framing_still_routes_to_the_topic(self):
        question = (
            "Explain Cell at an advanced level, including its formula and "
            "the main points to remember."
        )
        result = subject_rag.classify(question)
        self.assertIsNotNone(result)
        self.assertEqual((result.subject, result.topic), ("Biology", "Cell"))
        self.assertIn("Cell", subject_rag.answer(question))

    def test_advanced_answer_includes_mechanism_not_just_the_card(self):
        question = (
            "Explain the immune system at an advanced level, including its "
            "formula and main points."
        )
        answer = graph_rag.answer(question, [], []).lower()
        for part in ("mhc", "pattern-recognition", "memory cells", "no single universal equation"):
            with self.subTest(part=part):
                self.assertIn(part, answer)

    def test_advanced_routing_beats_short_curated_answers(self):
        cases = (
            (
                "Derive the Schur complement at an advanced level.",
                "schur complement - advanced explanation",
                "block gaussian elimination",
            ),
            (
                "Explain why nilpotent matrices have all eigenvalues zero.",
                "nilpotent matrix - advanced explanation",
                "strictly upper triangular",
            ),
        )
        for question, heading, mechanism in cases:
            with self.subTest(question=question):
                answer = graph_rag.answer(question, [], []).lower()
                self.assertIn(heading, answer)
                self.assertIn(mechanism, answer)

    def test_ideal_gas_equation_routes_to_its_own_card(self):
        result = subject_rag.classify("What is Ideal gas equation?")
        self.assertIsNotNone(result)
        self.assertEqual(
            (result.subject, result.topic),
            ("Chemistry", "Ideal gas equation"),
        )

    def test_skips_out_of_scope_and_flowchart_followups(self):
        self.assertIsNone(subject_rag.classify("What is a symphony?"))
        self.assertIsNone(subject_rag.classify("What is cricket?"))
        self.assertIsNone(subject_rag.classify("ok can u show me a process graph for this"))

    def test_python_answer_uses_subject_cards_before_wikipedia(self):
        cases = {
            "What is DNA?": ("double helix", "A pairs with T"),
            "Explain plate tectonics.": ("moving plates", "earthquakes"),
            "Explain a stack data structure.": ("LIFO", "push, pop"),
            "What is opportunity cost?": ("next best alternative",),
            "What is classical conditioning?": ("Pavlov", "neutral stimulus"),
        }
        for question, expected_parts in cases.items():
            with self.subTest(question=question):
                with patch.object(graph_rag, "wikipedia_answer") as lookup:
                    answer = graph_rag.answer(question, [], [])
                for part in expected_parts:
                    self.assertIn(part, answer)
                lookup.assert_not_called()


class SolverRoutingTests(unittest.TestCase):
    def test_solves_representative_math_physics_and_chemistry_questions(self):
        cases = {
            "Solve 2x + 3 = 11": "x = 4",
            "A car travels 20 m/s for 10 s. What distance?": "200 m",
            "A car starts from rest and accelerates at 2 m/s^2 for 5 s. Final velocity?": "10 m/s",
            "Force on a 10 kg mass accelerating at 3 m/s^2": "30 N",
            "Work done by 5 N force over 4 m": "20 J",
            "Power when 100 J of work is done in 5 s": "20 W",
            "Pressure for n=2 mol T=300 K V=10 L ideal gas": "atm",
            "Current through a 10 ohm resistor with 20 V": "2 A",
            "Moles in 18 g of water": "1 mol",
            "pH of 0.01 M HCl": "= 2",
            "How many moles of water form from 2 moles of H2 in 2H2 + O2 -> 2H2O": "2 mol water",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                result = subject_solvers.solve(question)
                self.assertIsNotNone(result)
                self.assertIn(expected, result)

    def test_python_answer_routes_solver_before_subject_cards(self):
        with patch.object(graph_rag, "wikipedia_answer") as lookup:
            answer = graph_rag.answer(
                "A car travels 20 m/s for 10 s. What distance?", [], []
            )
        self.assertIn("200 m", answer)
        lookup.assert_not_called()

    def test_equation_answer_names_the_solved_variable(self):
        answer = vido_math.handle_query("solve 2x+3=11")
        self.assertIsNotNone(answer)
        self.assertIn("x = 4", answer)


class ArxivEvidenceTests(unittest.TestCase):
    RESEARCH_QUESTION = "find recent papers on graph neural networks"

    def test_empty_search_cache_does_not_break_a_repeat_request(self):
        with (
            patch.dict(arxiv.CACHE, {}, clear=True),
            patch.object(arxiv.urllib.request, "urlopen", side_effect=OSError("offline")),
        ):
            first = arxiv.search("Monte Carlo methods", ("stat.CO",), limit=3)
            second = arxiv.search("Monte Carlo methods", ("stat.CO",), limit=3)

        self.assertEqual(first, [])
        self.assertEqual(second, [])

    def test_paper_answer_stays_badge_free(self):
        with (
            patch.object(arxiv, "search", side_effect=evaluation.offline_arxiv_search),
            patch.object(graph_rag, "wikipedia_answer", return_value=None),
        ):
            answer = graph_rag.answer(self.RESEARCH_QUESTION, [], [])

        self.assertIn("Graph neural networks for structured data", answer)
        lowered = answer.lower()
        for badge in (*evaluation.FORBIDDEN_BADGES, "arxiv", "wikipedia"):
            self.assertNotIn(badge.lower(), lowered)

    def test_paper_metadata_stays_in_the_internal_callback(self):
        seen = []

        def collect(question, paper_evidence):
            seen.append((question, paper_evidence))

        with (
            patch.object(arxiv, "search", side_effect=evaluation.offline_arxiv_search),
            patch.object(graph_rag, "wikipedia_answer", return_value=None),
        ):
            answer = graph_rag.answer(
                self.RESEARCH_QUESTION,
                [],
                [],
                arxiv_evidence_callback=collect,
            )

        self.assertIn("Graph neural networks for structured data", answer)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], self.RESEARCH_QUESTION)
        self.assertTrue(seen[0][1].papers)
        self.assertEqual(
            seen[0][1].papers[0].title,
            "Graph neural networks for structured data",
        )

    def test_explicit_research_prefers_papers_and_keeps_wikipedia_internal(self):
        seen = []

        def collect(question, answer_text):
            seen.append((question, answer_text))

        with (
            patch.object(graph_rag, "smart_answer", return_value=None),
            patch.object(graph_rag, "subject_answer", return_value=None),
            patch.object(graph_rag, "solve_subject_question", return_value=None),
            patch.object(graph_rag, "_world_fact_answer", return_value=None),
            patch.object(
                graph_rag,
                "wikipedia_answer",
                return_value="A clean encyclopedia answer.",
            ) as wiki,
            patch.object(graph_rag, "arxiv_answer", return_value="A paper answer") as papers,
        ):
            answer = graph_rag.answer(
                self.RESEARCH_QUESTION,
                [],
                [],
                evidence_callback=collect,
            )

        self.assertEqual(answer, "A paper answer")
        papers.assert_called_once()
        wiki.assert_not_called()
        # The visible answer is paper-backed, but the Wikipedia evidence check
        # still runs in parallel and never becomes a badge.
        self.assertEqual(seen, [(self.RESEARCH_QUESTION, "A paper answer")])

    def test_study_of_definitions_do_not_turn_into_paper_requests(self):
        self.assertFalse(graph_rag._wants_papers("What is the study of cells?"))
        self.assertTrue(graph_rag._wants_papers(self.RESEARCH_QUESTION))

    def test_research_request_words_do_not_become_the_topic(self):
        self.assertEqual(
            arxiv._topic(self.RESEARCH_QUESTION),
            "graph neural networks",
        )
        self.assertEqual(
            arxiv._topic("show me recent research on quantum field theory"),
            "quantum field theory",
        )

    def test_procedural_questions_do_not_search_papers(self):
        with patch.object(
            graph_rag,
            "arxiv_answer",
            side_effect=AssertionError("arXiv should not answer a procedure"),
        ) as papers:
            answer = graph_rag.answer("how to create a new database", [], [])

        self.assertIn("CREATE DATABASE", answer)
        papers.assert_not_called()
        self.assertFalse(arxiv.should_search("how to create a new database"))

    def test_categories_are_mapped_for_expanded_subjects(self):
        self.assertTrue(arxiv.categories_for("physics_branches"))
        self.assertTrue(arxiv.categories_for("research_papers"))
        self.assertEqual(arxiv.categories_for(""), ())


class WikipediaEvidenceTests(unittest.TestCase):
    def test_evidence_checks_subject_answers_without_adding_badges(self):
        seen = []

        def collect(question, answer_text):
            seen.append(
                wikipedia.evidence(question, answer_text, topic="DNA")
            )

        with (
            patch.dict(wikipedia.CACHE, {}, clear=True),
            patch.object(wikipedia, "_search", side_effect=evaluation.offline_search),
            patch.object(wikipedia, "_hf_tokens", side_effect=evaluation._regex_tokens),
            patch.object(graph_rag, "wikipedia_answer") as lookup,
        ):
            answer = graph_rag.answer(
                "What is DNA?", [], [], evidence_callback=collect
            )

        self.assertIn("double helix", answer)
        lookup.assert_not_called()
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].supported)
        self.assertEqual(seen[0].title, "DNA")
        for badge in evaluation.FORBIDDEN_BADGES:
            self.assertNotIn(badge, answer.lower())

    def test_wikipedia_fallback_answer_stays_badge_free(self):
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
            answer = graph_rag.answer("What is a symphony?", [], [])

        self.assertIn("extended musical composition", answer)
        for badge in evaluation.FORBIDDEN_BADGES:
            self.assertNotIn(badge, answer.lower())


class EvaluationHarnessTests(unittest.TestCase):
    def test_default_sweep_passes_all_offline_cases(self):
        report = evaluation.run_sweep()
        self.assertGreaterEqual(report.total, 20)
        self.assertEqual(report.passed, report.total)
        self.assertEqual(report.score, report.max_score)
        self.assertTrue(all(result.passed for result in report.results))


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


class ConceptGraphTests(unittest.TestCase):
    def test_one_shared_graph_has_partitions_and_cross_domain_links(self):
        stats = concept_graph.stats()
        self.assertEqual(stats["cards"], len(subject_rag.load_cards()))
        self.assertGreater(stats["edges"], 1000)
        self.assertGreater(stats["cross_domain_edges"], 100)
        self.assertGreater(stats["domains"], 10)
        self.assertEqual(
            len(concept_graph.partition("physics")),
            sum(1 for card in subject_rag.load_cards() if card.subject_key == "physics"),
        )

    def test_compare_uses_two_concepts_in_question_order(self):
        answer = graph_rag.answer(
            "Compare opportunity cost and comparative advantage", [], []
        )
        self.assertIn("Compare Opportunity cost and Comparative advantage", answer)
        self.assertIn("next best alternative", answer)
        self.assertIn("lower opportunity cost", answer)
        self.assertIn("Key difference", answer)

    def test_relation_answer_explores_a_cross_domain_path(self):
        answer = graph_rag.answer("How does entropy relate to information?", [], [])
        self.assertIn("Path:", answer)
        self.assertIn("Entropy", answer)
        self.assertIn("Information Theory Advanced", answer)
        self.assertIn("different partitions", answer)
        self.assertNotIn("Wikipedia", answer)
        self.assertNotIn("arXiv", answer)

    def test_mechanism_answer_uses_graph_evidence(self):
        answer = graph_rag.answer("Why does a catalyst not change equilibrium?", [], [])
        self.assertIn("Catalyst", answer)
        self.assertIn("equilibrium position", answer)
        self.assertIn("Connected idea", answer)

    def test_graph_definition_keeps_every_card_field_badge_free(self):
        answer = graph_rag.answer("What is DNA?", [], [])
        self.assertIn("double helix", answer)
        self.assertIn("A pairs with T", answer)
        self.assertIn("Connected concepts", answer)
        for badge in ("Python:", "Local:", "Engine:", "Source:", "Wikipedia:", "arXiv:"):
            self.assertNotIn(badge, answer)

    def test_unfamiliar_qualified_topic_still_uses_encyclopedia(self):
        with patch.object(
            graph_rag,
            "wikipedia_answer",
            return_value="Nash equilibrium is a stable state of a game.",
        ) as lookup:
            answer = graph_rag.answer("what is nash equilibrium", [], [])
        self.assertIn("Nash equilibrium", answer)
        lookup.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
