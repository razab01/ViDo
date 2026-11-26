"""Scored answer sweep for ViDo's Phase 2 answer engine.

Run with: python python/evaluation.py

The harness is offline by default. Wikipedia support checks use a small local
article set, and the answer text is checked for engine/source badges so a local
answer can never silently become a labelled chat message.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from unittest.mock import patch

import arxiv
import graph_rag
import wikipedia


FORBIDDEN_BADGES: Tuple[str, ...] = (
    "python:", "local:", "engine:", "source:", "wikipedia:", "arxiv:",
)

OFFLINE_WIKI: Dict[str, dict] = {
    "dna": {
        "title": "DNA",
        "extract": (
            "Deoxyribonucleic acid is a molecule that carries the genetic instructions used in "
            "the growth, functioning and reproduction of living organisms. Its double helix "
            "has two strands in which A pairs with T and C pairs with G."
        ),
        "rank": 0,
    },
    "plate tectonics": {
        "title": "Plate tectonics",
        "extract": (
            "Plate tectonics is the theory that Earth's outer shell is divided into moving "
            "plates. Their interactions cause earthquakes, volcanoes and the formation of "
            "mountain ranges."
        ),
        "rank": 0,
    },
    "opportunity cost": {
        "title": "Opportunity cost",
        "extract": (
            "Opportunity cost is the value of the next best alternative that is given up when "
            "a choice is made. It is a central idea in economics decision making."
        ),
        "rank": 0,
    },
    "symphony": {
        "title": "Symphony",
        "extract": (
            "A symphony is an extended musical composition in Western classical music, usually "
            "written for an orchestra. It is commonly structured in several movements with "
            "contrasting tempo and key."
        ),
        "rank": 0,
    },
}

OFFLINE_ARXIV: Tuple[dict, ...] = (
    {
        "id": "2401.00001",
        "title": "Graph neural networks for structured data",
        "summary": (
            "A survey of message passing, graph convolution and attention methods for learning "
            "on relational data. It compares node, edge and graph level prediction tasks."
        ),
        "published": "2024-01-02",
        "authors": ("A. Researcher",),
    },
    {
        "id": "2304.02643",
        "title": "Segment anything",
        "summary": (
            "A promptable image segmentation model and dataset designed to generalize across "
            "new images and tasks without retraining."
        ),
        "published": "2023-04-05",
        "authors": ("K. Kirillov",),
    },
)


@dataclass(frozen=True)
class EvalCase:
    name: str
    category: str
    question: str
    expected: Tuple[str, ...]
    rows: Tuple[dict, ...] = ()
    cols: Tuple[str, ...] = ()
    suggestions: Tuple[dict, ...] = ()
    documents: Tuple[dict, ...] = ()
    expect_wikipedia: bool = False
    evidence_topic: Optional[str] = None


@dataclass(frozen=True)
class EvalResult:
    name: str
    category: str
    answer: str
    passed: bool
    score: int
    max_score: int
    failures: Tuple[str, ...]


@dataclass(frozen=True)
class EvaluationReport:
    results: Tuple[EvalResult, ...]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def score(self) -> int:
        return sum(result.score for result in self.results)

    @property
    def max_score(self) -> int:
        return sum(result.max_score for result in self.results)

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "score": self.score,
            "max_score": self.max_score,
            "results": [
                {
                    "name": result.name,
                    "category": result.category,
                    "answer": result.answer,
                    "passed": result.passed,
                    "failures": list(result.failures),
                }
                for result in self.results
            ],
        }


DEFAULT_CASES: Tuple[EvalCase, ...] = (
    EvalCase(
        name="biology_card_dna",
        category="concept",
        question="What is DNA?",
        expected=("double helix", "A pairs with T"),
        evidence_topic="DNA",
    ),
    EvalCase(
        name="geography_card_plate_tectonics",
        category="concept",
        question="Explain plate tectonics.",
        expected=("moving plates", "earthquakes"),
        evidence_topic="plate tectonics",
    ),
    EvalCase(
        name="economics_card_opportunity_cost",
        category="concept",
        question="What is opportunity cost?",
        expected=("next best alternative",),
        evidence_topic="opportunity cost",
    ),
    EvalCase(
        name="math_concept_integration",
        category="concept",
        question="How is integration happen?",
        expected=("adding up many tiny pieces", "constant of integration"),
    ),
    EvalCase(
        name="math_solve_equation",
        category="solver",
        question="Solve 2x + 3 = 11",
        expected=("x = 4",),
    ),
    EvalCase(
        name="math_simplify_expression",
        category="solver",
        question="Simplify 2x + 3x",
        expected=("5*x",),
    ),
    EvalCase(
        name="probability_union_with_overlap",
        category="probability_statistics",
        question="If P(A)=0.4, P(B)=0.3 and P(A and B)=0.1, what is P(A or B)?",
        expected=("0.6", "P(A or B)"),
    ),
    EvalCase(
        name="probability_bayes_medical_test",
        category="probability_statistics",
        question=(
            "A disease affects 1% of people. A test is 99% sensitive and has a 5% "
            "false positive rate. If someone tests positive, what is the probability "
            "they have the disease?"
        ),
        expected=("0.1667", "Bayes theorem"),
    ),
    EvalCase(
        name="statistics_binomial_exact",
        category="probability_statistics",
        question=(
            "What is the probability of exactly 2 successes in 5 independent trials "
            "with success probability 0.4?"
        ),
        expected=("0.3456", "C(5,2)"),
    ),
    EvalCase(
        name="statistics_normal_z_score",
        category="probability_statistics",
        question=(
            "Heights are normal with mean 100 and standard deviation 15. "
            "What is the z-score for 110?"
        ),
        expected=("0.6667", "z ="),
    ),
    EvalCase(
        name="probability_multi_roll_sum",
        category="probability_statistics",
        question=(
            "what is probability of getting sum of 3 from six sided die rolled 3 times"
        ),
        expected=("1/216", "216"),
    ),
    EvalCase(
        name="conditional_coin_probability",
        category="probability_statistics",
        question=(
            "A fair coin is flipped 3 times. Given that at least one head occurred, "
            "what is the probability that exactly two heads occurred?"
        ),
        expected=("3/7", "Conditioning"),
    ),
    EvalCase(
        name="conditional_dice_first_face",
        category="probability_statistics",
        question=(
            "Two fair six-sided dice are rolled. Given that the first die shows 4, "
            "what is the probability that the sum is 10?"
        ),
        expected=("1/6", "fixed at 4"),
    ),
    EvalCase(
        name="conditional_dice_impossible_sum",
        category="probability_statistics",
        question=(
            "Two fair six-sided dice are rolled. Given that the first die shows 2, "
            "what is the probability that the sum is 10?"
        ),
        expected=("probability is 0", "0/6"),
    ),
    EvalCase(
        name="probability_two_roll_order",
        category="probability_statistics",
        question=(
            "A fair die is rolled twice. What is the probability that the second "
            "roll is strictly greater than the first?"
        ),
        expected=("5/12", "36"),
    ),
    EvalCase(
        name="probability_multi_roll_inequality",
        category="probability_statistics",
        question="A fair die is rolled 4 times. What is the probability that the sum is at least 20?",
        expected=("35/648", "1296"),
    ),
    EvalCase(
        name="probability_at_least_one_ace",
        category="probability_statistics",
        question=(
            "From a standard 52-card deck, two cards are drawn without replacement. "
            "What is the probability that at least one is an ace?"
        ),
        expected=("33/221", "C(48,2)"),
    ),
    EvalCase(
        name="probability_hypergeometric_defects",
        category="probability_statistics",
        question=(
            "In a batch of 100 items, 5 are defective. If 10 items are sampled "
            "without replacement, what is the probability that exactly 2 are defective?"
        ),
        expected=("0.0702", "hypergeometric"),
    ),
    EvalCase(
        name="probability_independence_test",
        category="probability_statistics",
        question=(
            "Suppose P(A) = 0.6, P(B) = 0.5, and P(A ∩ B) = 0.2. "
            "Are A and B independent?"
        ),
        expected=("No", "≠ 0.3"),
    ),
    EvalCase(
        name="probability_continuous_uniform_interval",
        category="probability_statistics",
        question="A random variable X is uniformly distributed on the interval [2, 8]. What is P(3 ≤ X ≤ 5)?",
        expected=("1/3", "Uniform[2, 8]"),
    ),
    EvalCase(
        name="statistics_binomial_sum",
        category="probability_statistics",
        question=(
            "If X has a binomial distribution with n = 10 and p = 0.3, "
            "what is P(X = 3) + P(X = 4)?"
        ),
        expected=("0.4669", "C(10,3)"),
    ),
    EvalCase(
        name="probability_birthday_problem",
        category="probability_statistics",
        question=(
            "Thirty people are in a room. Assuming 365 equally likely birthdays and "
            "no leap days, what is the probability that at least two share a birthday?"
        ),
        expected=("0.7063", "birthday"),
    ),
    EvalCase(
        name="probability_first_success_parity",
        category="probability_statistics",
        question=(
            "A fair coin is flipped until the first head appears. "
            "What is the probability that the number of flips is odd?"
        ),
        expected=("2/3", "geometric series"),
    ),
    EvalCase(
        name="probability_urn_same_colour",
        category="probability_statistics",
        question=(
            "A bag contains 3 red and 2 blue balls. Two balls are drawn with "
            "replacement. What is the probability that both are the same colour?"
        ),
        expected=("13/25", "same colour"),
    ),
    EvalCase(
        name="statistics_correlation_from_moments",
        category="probability_statistics",
        question=(
            "What is the correlation between X and Y if Cov(X,Y) = 6, "
            "Var(X) = 4, and Var(Y) = 9?"
        ),
        expected=("= 1", "Cov(X,Y)"),
    ),
    EvalCase(
        name="statistics_regression_intercept",
        category="probability_statistics",
        question="A regression line has slope 2.5 and passes through (4, 11). What is the intercept?",
        expected=("= 1", "2.5"),
    ),
    EvalCase(
        name="probability_independent_union",
        category="probability_statistics",
        question=(
            "For two independent events A and B, P(A) = 0.4 and P(B) = 0.7. "
            "What is P(A ∪ B)?"
        ),
        expected=("0.82", "independent"),
    ),
    EvalCase(
        name="probability_two_dice_sum_union",
        category="probability_statistics",
        question="Two fair six-sided dice are rolled. What is the probability that the sum is 7 or 11?",
        expected=("2/9", "0.2222"),
    ),
    EvalCase(
        name="probability_three_dice_equal",
        category="probability_statistics",
        question="Three fair six-sided dice are rolled. What is the probability that all three show the same face?",
        expected=("1/36", "0.0278"),
    ),
    EvalCase(
        name="probability_at_least_one_six",
        category="probability_statistics",
        question="A fair die is rolled 5 times. What is the probability of at least one six?",
        expected=("4651/7776", "0.5981"),
    ),
    EvalCase(
        name="conditional_dice_at_least_one_six",
        category="probability_statistics",
        question=(
            "Two fair six-sided dice are rolled. Given that at least one die shows a 6, "
            "what is the probability that the sum is 10?"
        ),
        expected=("2/11", "0.1818"),
    ),
    EvalCase(
        name="conditional_coin_first_flip",
        category="probability_statistics",
        question=(
            "A fair coin is flipped 5 times. Given that the first flip is heads, "
            "what is the probability of exactly 3 heads?"
        ),
        expected=("3/8", "0.375"),
    ),
    EvalCase(
        name="conditional_card_ace_spade",
        category="probability_statistics",
        question=(
            "From a standard 52-card deck, one card is drawn. Given that it is an ace, "
            "what is the probability that it is a spade?"
        ),
        expected=("1/4", "0.25"),
    ),
    EvalCase(
        name="probability_hypergeometric_at_least_two",
        category="probability_statistics",
        question=(
            "In a batch of 100 items, 5 are defective. If 10 items are sampled "
            "without replacement, what is the probability that at least 2 are defective?"
        ),
        expected=("0.0769", "hypergeometric"),
    ),
    EvalCase(
        name="probability_poisson_cdf",
        category="probability_statistics",
        question="If X has a Poisson distribution with mean 3, what is P(X ≤ 2)?",
        expected=("0.4232", "Poisson"),
    ),
    EvalCase(
        name="probability_exponential_cdf",
        category="probability_statistics",
        question="If X is exponential with rate 2, what is P(X < 1)?",
        expected=("0.8647", "exponential"),
    ),
    EvalCase(
        name="probability_normal_interval",
        category="probability_statistics",
        question=(
            "Heights are normally distributed with mean 100 and standard deviation 15. "
            "What is P(90 < X < 110)?"
        ),
        expected=("0.495", "z ="),
    ),
    EvalCase(
        name="statistics_binomial_tail",
        category="probability_statistics",
        question="If X ~ Binomial(20, 0.5), what is P(X ≥ 10)?",
        expected=("0.5881", "Binomial"),
    ),
    EvalCase(
        name="probability_urn_three_draws",
        category="probability_statistics",
        question=(
            "An urn has 4 red and 6 blue balls. Three balls are drawn without "
            "replacement. What is the probability of exactly 2 red balls?"
        ),
        expected=("3/10", "C(4,2)"),
    ),
    EvalCase(
        name="statistics_variance_transformation",
        category="probability_statistics",
        question="For a random variable X, Var(X) = 5. What is Var(2X + 3)?",
        expected=("= 20", "a²Var(X)"),
    ),
    EvalCase(
        name="statistics_slope_from_two_points",
        category="probability_statistics",
        question="A regression line passes through (1, 3) and (3, 7). What is its slope?",
        expected=("= 2", "y₂"),
    ),
    EvalCase(
        name="probability_consecutive_heads_waiting_time",
        category="probability_statistics",
        question=(
            "A fair coin is flipped until two consecutive heads appear. "
            "What is the expected number of flips?"
        ),
        expected=("expected number", "= 6"),
    ),
    EvalCase(
        name="probability_supplied_cdf_interval",
        category="probability_statistics",
        question="A random variable has CDF F(x) = 1 - e^(-2x) for x ≥ 0. What is P(1 < X ≤ 2)?",
        expected=("0.117", "F(2)"),
    ),
    EvalCase(
        name="probability_de_mere_comparison",
        category="probability_statistics",
        question=(
            "Which is more likely: getting at least one six in four rolls of a fair die, "
            "or getting at least one double six in 24 rolls of two fair dice?"
        ),
        expected=("0.5177", "0.4914"),
    ),
    EvalCase(
        name="probability_bayes_spam_filter",
        category="probability_statistics",
        question=(
            "A spam filter flags 90% of spam and 5% of legitimate mail. If 10% of mail "
            "is spam, what is the probability that a flagged message is actually spam?"
        ),
        expected=("2/3", "Bayes theorem"),
    ),
    EvalCase(
        name="rag_plan_card_joint_overlap",
        category="probability_statistics",
        question=(
            "Two cards are drawn from a standard 52-card deck without replacement. "
            "What is the probability that the first card is an ace and the second card is a spade?"
        ),
        expected=("1/52", "0.0192"),
    ),
    EvalCase(
        name="rag_plan_conditional_second_card",
        category="probability_statistics",
        question=(
            "The first card drawn from a standard 52-card deck is an ace. Without replacement, "
            "what is the probability that the second card is a spade?"
        ),
        expected=("1/4", "0.25"),
    ),
    EvalCase(
        name="rag_plan_reversed_dice_condition",
        category="probability_statistics",
        question=(
            "Two fair six-sided dice are rolled. Given that the sum is 8, what is the probability "
            "that the first die shows 5?"
        ),
        expected=("1/5", "0.2"),
    ),
    EvalCase(
        name="rag_plan_reversed_coin_condition",
        category="probability_statistics",
        question=(
            "A fair coin is flipped 4 times. Given that exactly two heads occurred, what is the "
            "probability that the first flip was heads?"
        ),
        expected=("1/2", "0.5"),
    ),
    EvalCase(
        name="rag_plan_pb_given_a",
        category="probability_statistics",
        question="Events A and B satisfy P(A and B) = 0.3 and P(A) = 0.5. What is P(B given A)?",
        expected=("0.6",),
    ),
    EvalCase(
        name="rag_plan_event_bayes_relation",
        category="probability_statistics",
        question=(
            "If P(A given B) = 0.6, P(B given A) = 0.4, and P(B) = 0.5, what is P(A)?"
        ),
        expected=("0.75",),
    ),
    EvalCase(
        name="rag_plan_total_probability",
        category="probability_statistics",
        question=(
            "For events A and B, P(A) = 0.6, P(B given A) = 0.5, and P(B given not A) = 0.2. "
            "What is P(B)?"
        ),
        expected=("0.38",),
    ),
    EvalCase(
        name="rag_plan_low_prevalence_bayes",
        category="probability_statistics",
        question=(
            "A disease affects 1 in 1000 people. A test is 99% sensitive and has a 5% false "
            "positive rate. If a randomly chosen person tests positive, what is the probability "
            "they have the disease?"
        ),
        expected=("0.0194",),
    ),
    EvalCase(
        name="rag_plan_exponential_memorylessness",
        category="probability_statistics",
        question="X is exponential with rate 2. What is P(X > 5 given X > 3)?",
        expected=("0.0183",),
    ),
    EvalCase(
        name="rag_plan_normal_upper_tail",
        category="probability_statistics",
        question=(
            "Heights are normally distributed with mean 100 and standard deviation 15. What is "
            "the probability that a randomly chosen height is greater than 110?"
        ),
        expected=("0.2525",),
    ),
    EvalCase(
        name="rag_plan_clt_sample_mean",
        category="probability_statistics",
        question=(
            "A population has mean 50 and standard deviation 10. For a random sample of size 25, "
            "approximate the probability that the sample mean exceeds 52."
        ),
        expected=("0.1587",),
    ),
    EvalCase(
        name="rag_plan_covariance_from_expectation",
        category="probability_statistics",
        question=(
            "For random variables X and Y, E[X] = 2, E[Y] = 3, and E[XY] = 8. What is Cov(X,Y)?"
        ),
        expected=("= 2",),
    ),
    EvalCase(
        name="rag_plan_variance_of_sum",
        category="probability_statistics",
        question=(
            "For random variables X and Y, Var(X) = 4, Var(Y) = 9, and Cov(X,Y) = 2. "
            "What is Var(X + Y)?"
        ),
        expected=("= 17",),
    ),
    EvalCase(
        name="rag_plan_die_race",
        category="probability_statistics",
        question="A fair die is rolled repeatedly. What is the probability that a 6 appears before a 5?",
        expected=("1/2",),
    ),
    EvalCase(
        name="rag_plan_coin_stopping_race",
        category="probability_statistics",
        question=(
            "A fair coin is flipped until either three heads or two tails appear. What is the "
            "probability that three heads appear first?"
        ),
        expected=("5/16", "0.3125"),
    ),
    EvalCase(
        name="rag_plan_counting_repetition",
        category="probability_statistics",
        question=(
            "How many four-digit strings can be formed from the digits 1, 2, 3, and 4 if "
            "repetition is allowed?"
        ),
        expected=("256",),
    ),
    EvalCase(
        name="rag_plan_hypergeom_more_than",
        category="probability_statistics",
        question=(
            "A batch has 20 items, of which 7 are defective. If 8 items are sampled without "
            "replacement, what is the probability that more than 2 are defective?"
        ),
        expected=("0.6084",),
    ),
    EvalCase(
        name="rag_plan_student_t_interval",
        category="probability_statistics",
        question=(
            "A sample of 16 has mean 50 and sample standard deviation 8. What is the 95% "
            "confidence interval for the population mean?"
        ),
        expected=("45.74", "54.26"),
    ),
    EvalCase(
        name="rag_plan_normal_within_wording",
        category="probability_statistics",
        question=(
            "For a normal distribution with mean 100 and standard deviation 15, what is the "
            "probability of being within 10 units of the mean?"
        ),
        expected=("0.495",),
    ),
    EvalCase(
        name="rag_plan_poisson_tail_wording",
        category="probability_statistics",
        question=(
            "Calls arrive according to a Poisson process with mean 3 per minute. What is the "
            "probability of at least 5 calls in a minute?"
        ),
        expected=("0.1847",),
    ),
    EvalCase(
        name="rag_plan_uniform_interval_wording",
        category="probability_statistics",
        question=(
            "X is uniformly distributed between 2 and 10. What is the probability that X is "
            "between 4 and 7?"
        ),
        expected=("3/8", "0.375"),
    ),
    EvalCase(
        name="rag_plan_exponential_survival_wording",
        category="probability_statistics",
        question=(
            "The waiting time is exponential with rate 2. What is the probability of waiting "
            "more than 3 units?"
        ),
        expected=("0.0025",),
    ),
    EvalCase(
        name="probability_urn_without_replacement",
        category="probability_statistics",
        question=(
            "An urn has 5 red and 3 blue balls. Two balls are drawn without replacement. "
            "What is the probability both are red?"
        ),
        expected=("5/14", "without replacement"),
    ),
    EvalCase(
        name="statistics_confidence_interval",
        category="probability_statistics",
        question=(
            "A sample of 100 has mean 50 and known population standard deviation 10. "
            "What is the 95% confidence interval for the population mean?"
        ),
        expected=("48.04", "51.96"),
    ),
    EvalCase(
        name="physics_kinematic_distance",
        category="solver",
        question="A car travels 20 m/s for 10 s. What distance?",
        expected=("200 m",),
    ),
    EvalCase(
        name="physics_final_velocity",
        category="solver",
        question="A car starts from rest and accelerates at 2 m/s^2 for 5 s. Final velocity?",
        expected=("10 m/s",),
    ),
    EvalCase(
        name="physics_newtons_second_law",
        category="solver",
        question="Force on a 10 kg mass accelerating at 3 m/s^2",
        expected=("30 N",),
    ),
    EvalCase(
        name="physics_work",
        category="solver",
        question="Work done by 5 N force over 4 m",
        expected=("20 J",),
    ),
    EvalCase(
        name="physics_power",
        category="solver",
        question="Power when 100 J of work is done in 5 s",
        expected=("20 W",),
    ),
    EvalCase(
        name="physics_ideal_gas_law",
        category="solver",
        question="Pressure for n=2 mol T=300 K V=10 L ideal gas",
        expected=("PV = nRT", "4.923 atm"),
    ),
    EvalCase(
        name="physics_ohms_law",
        category="solver",
        question="Current through a 10 ohm resistor with 20 V",
        expected=("2 A",),
    ),
    EvalCase(
        name="chemistry_moles_from_mass",
        category="solver",
        question="Moles in 18 g of water",
        expected=("1 mol",),
    ),
    EvalCase(
        name="chemistry_ph_strong_acid",
        category="solver",
        question="pH of 0.01 M HCl",
        expected=("-log(0.01)", "= 2"),
    ),
    EvalCase(
        name="chemistry_stoichiometry",
        category="solver",
        question="How many moles of water form from 2 moles of H2 in 2H2 + O2 -> 2H2O",
        expected=("2 mol water",),
    ),
    EvalCase(
        name="concept_graph_compare",
        category="concept_graph",
        question="Compare opportunity cost and comparative advantage",
        expected=("Compare Opportunity cost and Comparative advantage", "Key difference", "lower opportunity cost"),
    ),
    EvalCase(
        name="concept_graph_cross_domain_relation",
        category="concept_graph",
        question="How does entropy relate to information?",
        expected=("Path:", "Entropy", "Information Theory Advanced", "different partitions"),
    ),
    EvalCase(
        name="concept_graph_mechanism",
        category="concept_graph",
        question="Why does a catalyst not change equilibrium?",
        expected=("Catalyst", "equilibrium position", "Connected idea"),
    ),
    EvalCase(
        name="task_exact_record_lookup",
        category="task",
        question='"T09" tell me about thus task id',
        expected=("T09", "Submit report", "Due: 2026-11-30"),
        rows=(
            {"task_id": "T09", "task_text": "Submit report", "completed": "no",
             "due": "2026-11-30", "priority": "high"},
        ),
        cols=("task_id", "task_text", "completed", "due", "priority"),
    ),
    EvalCase(
        name="task_completed_records",
        category="task",
        question="which task are mostly completed?",
        expected=("Pay bills",),
        rows=(
            {"task_id": "T09", "task_text": "Submit report", "completed": "no"},
            {"task_id": "T10", "task_text": "Pay bills", "completed": "yes"},
        ),
        cols=("task_id", "task_text", "completed"),
    ),
    EvalCase(
        name="document_context_question",
        category="document",
        question="when does the production freeze start?",
        expected=('From "handbook.txt"', "2026-12-01"),
        documents=(
            {"name": "handbook.txt", "text": (
                "Retry policy. Use three attempts with exponential backoff.\n\n"
                "Deployment notes. The production freeze starts on 2026-12-01."
            )},
        ),
    ),
    EvalCase(
        name="out_of_scope_uses_wikipedia_cleanly",
        category="out_of_scope",
        question="What is a symphony?",
        expected=("extended musical composition",),
        expect_wikipedia=True,
        evidence_topic="symphony",
    ),
    EvalCase(
        name="physics_friction_planner",
        category="physics",
        question="A 10 kg block slides on a horizontal surface with kinetic friction coefficient 0.2. What is the friction force?",
        expected=("19.6", "N"),
    ),
    EvalCase(
        name="physics_inelastic_collision_planner",
        category="physics",
        question="A 2 kg object at 4 m/s sticks to a 3 kg object at rest. Find the final speed.",
        expected=("1.6", "m/s"),
    ),
    EvalCase(
        name="physics_projectile_planner",
        category="physics",
        question="A projectile is launched at 20 m/s at 30 degrees on level ground. What is the range?",
        expected=("35.348", "m"),
    ),
    EvalCase(
        name="physics_wave_planner",
        category="physics",
        question="A wave has frequency 50 Hz and wavelength 2 m. What is its speed?",
        expected=("100", "m/s"),
    ),
    EvalCase(
        name="physics_coulomb_planner",
        category="physics",
        question="What is the Coulomb force between charges of 2e-6 C and 3e-6 C separated by 0.5 m?",
        expected=("0.215712", "N"),
    ),
    EvalCase(
        name="physics_advanced_numeric_stays_exact",
        category="physics",
        question="Explain the friction force on a 10 kg block with kinetic friction coefficient 0.2 at an advanced level",
        expected=("19.6", "N"),
    ),
    EvalCase(
        name="physics_theory_string_theory",
        category="concept",
        question="What is String theory?",
        expected=("strings", "vibration modes"),
    ),
    EvalCase(
        name="physics_theory_advanced_field_theory",
        category="concept",
        question="Explain Quantum Field Theory at an advanced level, including its formula and the main points to remember.",
        expected=("fields", "excitations"),
    ),
    EvalCase(
        name="physics_theory_computational_methods",
        category="concept",
        question="What is Computational Physics Methods?",
        expected=("numerical methods", "simulations"),
    ),
)


def _regex_tokens(text: str) -> List[str]:
    return [word.lower() for word in re.findall(r"[a-z0-9]+", str(text or "").lower()) if word]


def offline_search(topic: str, limit: int = 10) -> List[dict]:
    """Return local article candidates for deterministic evidence checks."""
    key = wikipedia._normalize_title(topic)
    matches: List[dict] = []
    for article_key, article in OFFLINE_WIKI.items():
        if key == article_key or article_key in key or key in article_key:
            matches.append(dict(article))
    return matches[:limit]


def offline_wikipedia_answer(question: str) -> Optional[str]:
    topic = wikipedia.topic_phrase(question)
    candidates = offline_search(topic)
    extract = wikipedia._best_extract(question, topic, candidates)
    return wikipedia._trim(extract) if extract else None


def offline_arxiv_search(
    topic: str,
    categories: Sequence[str] = (),
    limit: int = 3,
) -> List[dict]:
    """Return deterministic paper metadata without touching the network."""
    key = _normalize_for_match(topic)
    matched = [
        dict(paper) for paper in OFFLINE_ARXIV
        if not key or any(word in key for word in paper["title"].lower().split())
    ]
    return (matched or [dict(paper) for paper in OFFLINE_ARXIV])[:max(1, min(limit, 5))]


def _normalize_for_match(text: str) -> str:
    return " ".join(word.lower() for word in re.findall(r"[a-z0-9]+", str(text or "")))


def _score_case(
    case: EvalCase,
    answer: str,
    wiki_calls: Sequence[str],
    evidence: Optional[wikipedia.Evidence],
) -> EvalResult:
    lowered = answer.lower()
    failures: List[str] = []
    score = 0
    max_score = len(case.expected) + 2 + (1 if case.evidence_topic else 0)

    for part in case.expected:
        if part.lower() in lowered:
            score += 1
        else:
            failures.append(f"missing expected text: {part}")

    if any(badge in lowered for badge in FORBIDDEN_BADGES):
        failures.append("answer contains an engine/source badge")
    else:
        score += 1

    if bool(wiki_calls) == case.expect_wikipedia:
        score += 1
    else:
        failures.append(
            f"wikipedia expected={case.expect_wikipedia}, calls={len(wiki_calls)}"
        )

    if case.evidence_topic:
        if evidence is not None and evidence.supported:
            score += 1
        else:
            failures.append(f"unsupported Wikipedia evidence for {case.evidence_topic}")

    return EvalResult(
        name=case.name,
        category=case.category,
        answer=answer,
        passed=not failures,
        score=score,
        max_score=max_score,
        failures=tuple(failures),
    )


def run_sweep(
    cases: Optional[Sequence[EvalCase]] = None,
    wikipedia_lookup: Optional[Callable[[str], Optional[str]]] = None,
    search: Optional[Callable[[str, int], List[dict]]] = None,
) -> EvaluationReport:
    """Run the scored sweep. Defaults are fully offline and deterministic."""
    selected = list(DEFAULT_CASES if cases is None else cases)
    lookup = wikipedia_lookup or offline_wikipedia_answer
    search_fn = search or offline_search
    results: List[EvalResult] = []

    with (
        patch.dict(wikipedia.CACHE, {}, clear=True),
        patch.dict(arxiv.CACHE, {}, clear=True),
        patch.object(wikipedia, "_search", side_effect=search_fn),
        patch.object(wikipedia, "_hf_tokens", side_effect=_regex_tokens),
        patch.object(arxiv, "search", side_effect=offline_arxiv_search),
    ):
        for case in selected:
            wiki_calls: List[str] = []
            evidence: Dict[str, wikipedia.Evidence] = {}

            def tracked_lookup(question: str) -> Optional[str]:
                wiki_calls.append(question)
                return lookup(question)

            def collect_evidence(question: str, answer_text: str) -> None:
                evidence[case.name] = wikipedia.evidence(
                    question,
                    answer_text,
                    topic=case.evidence_topic,
                )

            with patch.object(graph_rag, "wikipedia_answer", side_effect=tracked_lookup):
                answer = graph_rag.answer(
                    case.question,
                    list(case.rows),
                    list(case.cols),
                    suggestions=list(case.suggestions),
                    documents=list(case.documents),
                    evidence_callback=collect_evidence,
                )

            results.append(_score_case(
                case,
                answer,
                wiki_calls,
                evidence.get(case.name),
            ))

    return EvaluationReport(tuple(results))


def main() -> int:
    report = run_sweep()
    print(
        f"Phase 2 sweep: {report.passed}/{report.total} cases passed; "
        f"score {report.score}/{report.max_score}"
    )
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} [{result.category}] {result.name}")
        for failure in result.failures:
            print(f"     - {failure}")
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
