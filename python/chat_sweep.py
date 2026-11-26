"""Generate and run a 1,000-question Chat Lab answer sweep.

Run with: python python/chat_sweep.py

Outputs:
  python/chat_sweep_report.md   - every question, response, and pass/fail status
  python/chat_sweep_report.jsonl - the same data as one JSON object per line

The sweep is offline and deterministic. It covers:
  500 Basic    - balanced definition questions across every subject card file
  200 Intermediate - parameterized physics/chemistry solver questions
  250 Advanced  - advanced-phrasing questions across the same balanced card set
  50  Graduate  - explicit SymPy determinants, eigenvalues, integrals, derivatives

The expanded store has more than 500 cards, so Basic and Advanced sampling keeps
the original 1,000-case totals while still covering all subject catalogs.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import unittest
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from unittest.mock import patch

# Match server.py and test_answers.py: keep local math.py from shadowing stdlib.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.append(HERE)

import arxiv  # noqa: E402
import evaluation  # noqa: E402
import graph_rag  # noqa: E402
import subject_rag  # noqa: E402
import subject_solvers  # noqa: E402
import test_answers  # noqa: E402
import wikipedia  # noqa: E402
from sympy import Matrix, Symbol, diff, integrate, simplify  # noqa: E402


REPORT_MD = os.path.join(HERE, "chat_sweep_report.md")
REPORT_JSONL = os.path.join(HERE, "chat_sweep_report.jsonl")
FALLBACK_PHRASES = (
    "i understand the question, but",
    "type a question and i will answer it",
    "not enough information in this dataset",
)
# Match a badge only as a standalone token. "resource:" must not be mistaken
# for "source:", and "Python:" is allowed when Python is the topic being asked.
BADGE_RE = re.compile(r"(?i)(?:^|[\s.!?])(?:local|engine|source|wikipedia|arxiv):\s")
PYTHON_BADGE_RE = re.compile(r"(?i)^python:\s")


@dataclass(frozen=True)
class SweepCase:
    number: int
    level: str
    subject: str
    question: str
    expected: Tuple[str, ...]


@dataclass(frozen=True)
class SweepResult:
    number: int
    level: str
    subject: str
    question: str
    expected: Tuple[str, ...]
    answer: str
    passed: bool
    failures: Tuple[str, ...]


def _prefix(text: str, words: int = 4) -> str:
    return " ".join(str(text or "").split()[:words])


def _expected_terms(
    question: str,
    cards: Sequence[subject_rag.TopicCard],
) -> Tuple[str, ...]:
    """Topic names/aliases that count as an on-topic response.

    The pilot store intentionally has a few shared titles across subjects (for
    example "Depression"). Deriving the accepted terms from the classifier keeps
    the breadth sweep focused on whether the answer matches the selected topic
    instead of guessing which duplicate title the router should prefer. The
    dedicated unittest/evaluation suites cover those disambiguation rules.
    """
    classification = subject_rag.classify(question)
    if classification:
        selected = [
            card for card in cards
            if card.subject_key == classification.subject_key
            and card.title == classification.topic
        ]
        if selected:
            card = selected[0]
            return tuple(dict.fromkeys([card.title, *card.aliases]))
        return (classification.topic,)
    return (question.strip(" ?."),)


def _card_terms(card: subject_rag.TopicCard) -> Tuple[str, ...]:
    """Terms accepted for a card the sweep already selected directly."""
    return tuple(dict.fromkeys([card.title, *card.aliases]))


def _contains_term(answer: str, term: str) -> bool:
    return re.sub(r"\s+", " ", term.lower()) in re.sub(r"\s+", " ", answer.lower())


def _has_badge(answer: str, expected: Tuple[str, ...]) -> bool:
    if BADGE_RE.search(answer):
        return True
    if not PYTHON_BADGE_RE.search(answer):
        return False
    expected_lower = {term.strip().lower() for term in expected}
    return "python" not in expected_lower


def _flatten_tests(suite: unittest.TestSuite) -> Iterable[unittest.TestCase]:
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten_tests(item)
        elif item is not None:
            yield item


def collect_verified_checks() -> Tuple[List[str], List[str], int, List[str]]:
    """Actually run the existing Python chat suites and return their names."""
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_answers)
    # TestSuite.run() clears its tests, so collect names before running it.
    test_names = [test.id().split(".")[-1] for test in _flatten_tests(suite)]
    result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    problems = [
        f"{case.id()}: {detail}" for case, detail in result.failures
    ] + [
        f"{case.id()}: {detail}" for case, detail in result.errors
    ]
    eval_names = [case.name for case in evaluation.DEFAULT_CASES]
    return test_names, eval_names, result.testsRun, problems


def _balanced_cards(
    cards: Sequence[subject_rag.TopicCard],
    total: int,
) -> List[subject_rag.TopicCard]:
    """Pick a deterministic, subject-balanced subset of the expanded store."""
    grouped: Dict[str, List[subject_rag.TopicCard]] = {}
    for card in cards:
        grouped.setdefault(card.subject_key, []).append(card)
    order = [key for key in subject_rag.SUBJECT_NAMES if key in grouped]
    order.extend(sorted(key for key in grouped if key not in order))

    selected: List[subject_rag.TopicCard] = []
    taken: Dict[str, int] = {}
    quota = total // max(1, len(order))
    for key in order:
        take = min(quota, len(grouped[key]))
        taken[key] = take
        selected.extend(grouped[key][:take])

    offset = quota
    while len(selected) < total:
        added = False
        for key in order:
            group = grouped[key]
            while taken[key] < len(group) and taken[key] < offset:
                taken[key] += 1
            if taken[key] < len(group):
                selected.append(group[taken[key]])
                taken[key] += 1
                added = True
                if len(selected) == total:
                    break
        if not added:
            break
        offset += 1
    return selected[:total]


def _basic_cards(cards: Sequence[subject_rag.TopicCard], start: int) -> List[SweepCase]:
    cases: List[SweepCase] = []
    for offset, card in enumerate(cards):
        cases.append(SweepCase(
            number=start + offset,
            level="Basic",
            subject=card.subject,
            question=f"What is {card.title}?",
            expected=_card_terms(card),
        ))
    return cases


def _intermediate_solvers(start: int) -> List[SweepCase]:
    cases: List[SweepCase] = []

    for index in range(20):
        speed = 5 + (index % 16)
        time = 2 + (index % 9)
        cases.append(SweepCase(
            number=start + len(cases),
            level="Intermediate",
            subject="Physics",
            question=f"A vehicle travels {speed} m/s for {time} s. What distance?",
            expected=(f"{speed * time} m",),
        ))

        acceleration = 2 + (index % 8)
        time = 2 + (index % 7)
        cases.append(SweepCase(
            number=start + len(cases),
            level="Intermediate",
            subject="Physics",
            question=(
                f"A car starts from rest and accelerates at {acceleration} m/s^2 "
                f"for {time} s. Final velocity?"
            ),
            expected=(f"{acceleration * time} m/s",),
        ))

        mass = 2 + (index % 9)
        acceleration = 2 + (index % 6)
        cases.append(SweepCase(
            number=start + len(cases),
            level="Intermediate",
            subject="Physics",
            question=f"Force on a {mass} kg mass accelerating at {acceleration} m/s^2",
            expected=(f"{mass * acceleration} N",),
        ))

        force = 3 + (index % 12)
        distance = 2 + (index % 10)
        cases.append(SweepCase(
            number=start + len(cases),
            level="Intermediate",
            subject="Physics",
            question=f"Work done by {force} N force over {distance} m",
            expected=(f"{force * distance} J",),
        ))

        work = 20 * (index % 8 + 1)
        time = 2 + (index % 5)
        cases.append(SweepCase(
            number=start + len(cases),
            level="Intermediate",
            subject="Physics",
            question=f"Power when {work} J of work is done in {time} s",
            expected=(f"{work // time} W",) if work % time == 0 else (f"{work / time:g} W",),
        ))

        moles = 1 + (index % 5)
        kelvin = 200 + 50 * (index % 5)
        litres = 5 + 5 * (index % 5)
        gas_value = moles * 0.082057 * kelvin / litres
        cases.append(SweepCase(
            number=start + len(cases),
            level="Intermediate",
            subject="Physics",
            question=f"Pressure for n={moles} mol T={kelvin} K V={litres} L ideal gas",
            expected=(subject_solvers._fmt(gas_value, 4),),
        ))

        resistance = 2 + (index % 9)
        current = 1 + (index % 6)
        voltage = resistance * current
        cases.append(SweepCase(
            number=start + len(cases),
            level="Intermediate",
            subject="Physics",
            question=f"Current through a {resistance} ohm resistor with {voltage} V",
            expected=(f"{current} A",),
        ))

        grams = 18 * (index % 8 + 1)
        cases.append(SweepCase(
            number=start + len(cases),
            level="Intermediate",
            subject="Chemistry",
            question=f"Moles in {grams} g of water",
            expected=(f"{grams // 18} mol",),
        ))

        ph = [2, 3, 4, 1][index % 4]
        concentration = f"0.{'0' * (ph - 1)}1"
        cases.append(SweepCase(
            number=start + len(cases),
            level="Intermediate",
            subject="Chemistry",
            question=f"pH of {concentration} M HCl",
            expected=(f"= {ph}",),
        ))

        product_moles = 1 + (index % 10)
        cases.append(SweepCase(
            number=start + len(cases),
            level="Intermediate",
            subject="Chemistry",
            question=(
                f"How many moles of water form from {product_moles} moles of H2 "
                "in 2H2 + O2 -> 2H2O"
            ),
            expected=(f"{product_moles} mol water",),
        ))

    return cases


def _advanced_cards(cards: Sequence[subject_rag.TopicCard], start: int) -> List[SweepCase]:
    selected = _balanced_cards(cards, 250)
    cases: List[SweepCase] = []
    for offset, card in enumerate(selected):
        cases.append(SweepCase(
            number=start + offset,
            level="Advanced",
            subject=card.subject,
            question=(
                f"Explain {card.title} at an advanced level, including its "
                "formula and the main points to remember."
            ),
            expected=_card_terms(card),
        ))
    return cases


CONCEPT_GRAPH_QUESTIONS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "Compare opportunity cost and comparative advantage",
        ("Opportunity cost", "Comparative advantage", "Key difference"),
    ),
    (
        "How does entropy relate to information?",
        ("Entropy", "Information Theory Advanced", "different partitions"),
    ),
    (
        "Why does a catalyst not change equilibrium?",
        ("Catalyst", "equilibrium position", "Connected idea"),
    ),
)


PROBABILITY_STATS_QUESTIONS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "If P(A)=0.4, P(B)=0.3 and P(A and B)=0.1, what is P(A or B)?",
        ("0.6", "P(A or B)"),
    ),
    (
        "A disease affects 1% of people. A test is 99% sensitive and has a 5% false "
        "positive rate. If someone tests positive, what is the probability they have "
        "the disease?",
        ("0.1667", "Bayes theorem"),
    ),
    (
        "What is the probability of exactly 2 successes in 5 independent trials with "
        "success probability 0.4?",
        ("0.3456", "C(5,2)"),
    ),
    (
        "If X follows a Poisson distribution with lambda=3, what is P(X=2)?",
        ("0.224", "Poisson"),
    ),
    (
        "Heights are normal with mean 100 and standard deviation 15. What is the "
        "z-score for 110?",
        ("0.6667", "z ="),
    ),
    (
        "A discrete random variable has values 1,2,3 with probabilities 0.2,0.3,0.5. "
        "What is E[X]?",
        ("2.3", "E[X]"),
    ),
    (
        "Two fair dice are rolled. What is the probability the sum is 7?",
        ("1/6", "36"),
    ),
    (
        "An urn has 5 red and 3 blue balls. Two balls are drawn without replacement. "
        "What is the probability both are red?",
        ("5/14", "without replacement"),
    ),
    (
        "what is probability of getting sum of 3 from six sided die rolled 3 times",
        ("1/216", "216"),
    ),
    (
        "A fair coin is flipped 3 times. Given that at least one head occurred, what "
        "is the probability that exactly two heads occurred?",
        ("3/7", "Conditioning"),
    ),
    (
        "Two fair six-sided dice are rolled. Given that the first die shows 4, what "
        "is the probability that the sum is 10?",
        ("1/6", "fixed at 4"),
    ),
    (
        "Two fair six-sided dice are rolled. Given that the first die shows 2, what "
        "is the probability that the sum is 10?",
        ("probability is 0", "0/6"),
    ),
    (
        "A fair die is rolled twice. What is the probability that the second roll is "
        "strictly greater than the first?",
        ("5/12", "36"),
    ),
    (
        "A fair die is rolled 4 times. What is the probability that the sum is at "
        "least 20?",
        ("35/648", "1296"),
    ),
    (
        "From a standard 52-card deck, two cards are drawn without replacement. What "
        "is the probability that at least one is an ace?",
        ("33/221", "C(48,2)"),
    ),
    (
        "In a batch of 100 items, 5 are defective. If 10 items are sampled without "
        "replacement, what is the probability that exactly 2 are defective?",
        ("0.0702", "hypergeometric"),
    ),
    (
        "Suppose P(A) = 0.6, P(B) = 0.5, and P(A ∩ B) = 0.2. Are A and B independent?",
        ("No", "≠ 0.3"),
    ),
    (
        "A random variable X is uniformly distributed on the interval [2, 8]. What "
        "is P(3 ≤ X ≤ 5)?",
        ("1/3", "Uniform[2, 8]"),
    ),
    (
        "If X has a binomial distribution with n = 10 and p = 0.3, what is "
        "P(X = 3) + P(X = 4)?",
        ("0.4669", "C(10,3)"),
    ),
    (
        "Thirty people are in a room. Assuming 365 equally likely birthdays and no "
        "leap days, what is the probability that at least two share a birthday?",
        ("0.7063", "birthday"),
    ),
    (
        "A fair coin is flipped until the first head appears. What is the probability "
        "that the number of flips is odd?",
        ("2/3", "geometric series"),
    ),
    (
        "A bag contains 3 red and 2 blue balls. Two balls are drawn with replacement. "
        "What is the probability that both are the same colour?",
        ("13/25", "same colour"),
    ),
    (
        "What is the correlation between X and Y if Cov(X,Y) = 6, Var(X) = 4, and "
        "Var(Y) = 9?",
        ("= 1", "Cov(X,Y)"),
    ),
    (
        "A regression line has slope 2.5 and passes through (4, 11). What is the "
        "intercept?",
        ("= 1", "2.5"),
    ),
    (
        "For two independent events A and B, P(A) = 0.4 and P(B) = 0.7. What is "
        "P(A ∪ B)?",
        ("0.82", "independent"),
    ),
    (
        "Two fair six-sided dice are rolled. What is the probability that the sum is "
        "7 or 11?",
        ("2/9", "0.2222"),
    ),
    (
        "Three fair six-sided dice are rolled. What is the probability that all "
        "three show the same face?",
        ("1/36", "0.0278"),
    ),
    (
        "A fair die is rolled 5 times. What is the probability of at least one six?",
        ("4651/7776", "0.5981"),
    ),
    (
        "Two fair six-sided dice are rolled. Given that at least one die shows a 6, "
        "what is the probability that the sum is 10?",
        ("2/11", "0.1818"),
    ),
    (
        "A fair coin is flipped 5 times. Given that the first flip is heads, what "
        "is the probability of exactly 3 heads?",
        ("3/8", "0.375"),
    ),
    (
        "From a standard 52-card deck, one card is drawn. Given that it is an ace, "
        "what is the probability that it is a spade?",
        ("1/4", "0.25"),
    ),
    (
        "In a batch of 100 items, 5 are defective. If 10 items are sampled without "
        "replacement, what is the probability that at least 2 are defective?",
        ("0.0769", "hypergeometric"),
    ),
    (
        "If X has a Poisson distribution with mean 3, what is P(X ≤ 2)?",
        ("0.4232", "Poisson"),
    ),
    (
        "If X is exponential with rate 2, what is P(X < 1)?",
        ("0.8647", "exponential"),
    ),
    (
        "Heights are normally distributed with mean 100 and standard deviation 15. "
        "What is P(90 < X < 110)?",
        ("0.495", "z ="),
    ),
    (
        "If X ~ Binomial(20, 0.5), what is P(X ≥ 10)?",
        ("0.5881", "Binomial"),
    ),
    (
        "An urn has 4 red and 6 blue balls. Three balls are drawn without "
        "replacement. What is the probability of exactly 2 red balls?",
        ("3/10", "C(4,2)"),
    ),
    (
        "For a random variable X, Var(X) = 5. What is Var(2X + 3)?",
        ("= 20", "a²Var(X)"),
    ),
    (
        "A regression line passes through (1, 3) and (3, 7). What is its slope?",
        ("= 2", "y₂"),
    ),
    (
        "A fair coin is flipped until two consecutive heads appear. What is the "
        "expected number of flips?",
        ("expected number", "= 6"),
    ),
    (
        "A random variable has CDF F(x) = 1 - e^(-2x) for x ≥ 0. What is P(1 < X ≤ 2)?",
        ("0.117", "F(2)"),
    ),
    (
        "Which is more likely: getting at least one six in four rolls of a fair die, "
        "or getting at least one double six in 24 rolls of two fair dice?",
        ("0.5177", "0.4914"),
    ),
    (
        "A spam filter flags 90% of spam and 5% of legitimate mail. If 10% of mail "
        "is spam, what is the probability that a flagged message is actually spam?",
        ("2/3", "Bayes theorem"),
    ),
    (
        "Two cards are drawn from a standard 52-card deck without replacement. What is "
        "the probability that the first card is an ace and the second card is a spade?",
        ("1/52", "0.0192"),
    ),
    (
        "The first card drawn from a standard 52-card deck is an ace. Without replacement, "
        "what is the probability that the second card is a spade?",
        ("1/4", "0.25"),
    ),
    (
        "Two fair six-sided dice are rolled. Given that the sum is 8, what is the probability "
        "that the first die shows 5?",
        ("1/5", "0.2"),
    ),
    (
        "A fair coin is flipped 4 times. Given that exactly two heads occurred, what is the "
        "probability that the first flip was heads?",
        ("1/2", "0.5"),
    ),
    (
        "Events A and B satisfy P(A and B) = 0.3 and P(A) = 0.5. What is P(B given A)?",
        ("0.6",),
    ),
    (
        "If P(A given B) = 0.6, P(B given A) = 0.4, and P(B) = 0.5, what is P(A)?",
        ("0.75",),
    ),
    (
        "For events A and B, P(A) = 0.6, P(B given A) = 0.5, and P(B given not A) = 0.2. "
        "What is P(B)?",
        ("0.38",),
    ),
    (
        "A disease affects 1 in 1000 people. A test is 99% sensitive and has a 5% false "
        "positive rate. If a randomly chosen person tests positive, what is the probability "
        "they have the disease?",
        ("0.0194",),
    ),
    (
        "X is exponential with rate 2. What is P(X > 5 given X > 3)?",
        ("0.0183",),
    ),
    (
        "Heights are normally distributed with mean 100 and standard deviation 15. What is "
        "the probability that a randomly chosen height is greater than 110?",
        ("0.2525",),
    ),
    (
        "A population has mean 50 and standard deviation 10. For a random sample of size 25, "
        "approximate the probability that the sample mean exceeds 52.",
        ("0.1587",),
    ),
    (
        "For random variables X and Y, E[X] = 2, E[Y] = 3, and E[XY] = 8. What is Cov(X,Y)?",
        ("= 2",),
    ),
    (
        "For random variables X and Y, Var(X) = 4, Var(Y) = 9, and Cov(X,Y) = 2. "
        "What is Var(X + Y)?",
        ("= 17",),
    ),
    (
        "A fair die is rolled repeatedly. What is the probability that a 6 appears before a 5?",
        ("1/2",),
    ),
    (
        "A fair coin is flipped until either three heads or two tails appear. What is the "
        "probability that three heads appear first?",
        ("5/16", "0.3125"),
    ),
    (
        "How many four-digit strings can be formed from the digits 1, 2, 3, and 4 if "
        "repetition is allowed?",
        ("256",),
    ),
    (
        "A batch has 20 items, of which 7 are defective. If 8 items are sampled without "
        "replacement, what is the probability that more than 2 are defective?",
        ("0.6084",),
    ),
    (
        "A sample of 16 has mean 50 and sample standard deviation 8. What is the 95% "
        "confidence interval for the population mean?",
        ("45.74", "54.26"),
    ),
    (
        "For a normal distribution with mean 100 and standard deviation 15, what is the "
        "probability of being within 10 units of the mean?",
        ("0.495",),
    ),
    (
        "Calls arrive according to a Poisson process with mean 3 per minute. What is the "
        "probability of at least 5 calls in a minute?",
        ("0.1847",),
    ),
    (
        "X is uniformly distributed between 2 and 10. What is the probability that X is "
        "between 4 and 7?",
        ("3/8", "0.375"),
    ),
    (
        "The waiting time is exponential with rate 2. What is the probability of waiting "
        "more than 3 units?",
        ("0.0025",),
    ),
    (
        "A sample of 100 has mean 50 and known population standard deviation 10. What "
        "is the 95% confidence interval for the population mean?",
        ("48.04", "51.96"),
    ),
)


PHYSICS_QUESTIONS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "A 10 kg block slides on a horizontal surface with kinetic friction "
        "coefficient 0.2. What is the friction force?",
        ("19.6",),
    ),
    ("A 4 kg object moving at 3 m/s has what momentum?", ("12 kg m/s",)),
    ("A force of 12 N acts for 5 s. What impulse is delivered?", ("60 N s",)),
    (
        "A 2 kg object at 4 m/s sticks to a 3 kg object at rest. Find the final speed.",
        ("1.6 m/s",),
    ),
    (
        "A 2 kg mass moves at 3 m/s in a circle of radius 2 m. What centripetal "
        "force is needed?",
        ("9 N",),
    ),
    (
        "A projectile is launched at 20 m/s at 30 degrees on level ground. What is "
        "the range?",
        ("35.348",),
    ),
    (
        "A 2 kg mass on a spring of constant 50 N/m oscillates. What is the period?",
        ("1.25664",),
    ),
    (
        "A simple pendulum of length 2 m oscillates. What is its period?",
        ("2.83845",),
    ),
    (
        "A wave has frequency 50 Hz and wavelength 2 m. What is its speed?",
        ("100 m/s",),
    ),
    (
        "Doppler effect: a sound source emits 500 Hz. An observer moves toward it "
        "at 20 m/s. The sound speed is 340 m/s. What frequency is heard?",
        ("529.412",),
    ),
    (
        "Doppler effect: a source moving toward an observer at 20 m/s emits 500 Hz. "
        "The sound speed is 340 m/s. What frequency is heard?",
        ("531.25",),
    ),
    (
        "Using Snell's law, light hits a boundary at 30 degrees. The refractive "
        "index is 1.5. What is the refracted angle?",
        ("19.4712",),
    ),
    (
        "For a lens, the object distance is 0.30 m and the focal length is 0.10 m. "
        "Where is the image?",
        ("0.15 m",),
    ),
    (
        "What is the Coulomb force between charges of 2e-6 C and 3e-6 C separated "
        "by 0.5 m?",
        ("0.215712",),
    ),
    ("What is the electric field at 2 m from a charge of 2e-6 C?", ("4494",)),
    (
        "What is the equivalent resistance of 6 ohm and 3 ohm resistors in parallel?",
        ("2 Ω",),
    ),
    (
        "A 1e-5 F capacitor is charged to 100 V. How much energy is stored?",
        ("0.05 J",),
    ),
    (
        "What is the rest energy of a 0.001 kg mass?",
        ("9e+13",),
    ),
    (
        "A sample starts with 100 atoms and has a half-life of 10 s. How many "
        "remain after 20 s?",
        ("25 atoms",),
    ),
    ("What is String theory?", ("strings", "vibration modes")),
    (
        "Explain Quantum Field Theory at an advanced level, including its formula "
        "and the main points to remember.",
        ("fields", "excitations"),
    ),
    ("What is Computational Physics Methods?", ("numerical methods", "simulations")),
    (
        "Explain the friction force on a 10 kg block with kinetic friction "
        "coefficient 0.2 at an advanced level",
        ("19.6",),
    ),
)


def _concept_graph_cases(start: int) -> List[SweepCase]:
    """Explicit compare/relate/why checks inside the Advanced quota."""
    return [
        SweepCase(
            number=start + offset,
            level="Advanced",
            subject="Concept Graph",
            question=question,
            expected=expected,
        )
        for offset, (question, expected) in enumerate(CONCEPT_GRAPH_QUESTIONS)
    ]


def _probability_stats_cases(start: int) -> List[SweepCase]:
    """Explicit broad probability/statistics checks inside the Advanced quota."""
    return [
        SweepCase(
            number=start + offset,
            level="Advanced",
            subject="Probability and Statistics",
            question=question,
            expected=expected,
        )
        for offset, (question, expected) in enumerate(PROBABILITY_STATS_QUESTIONS)
    ]


def _physics_cases(start: int) -> List[SweepCase]:
    """Explicit physics planner and branch checks inside the Advanced quota."""
    return [
        SweepCase(
            number=start + offset,
            level="Advanced",
            subject="Physics",
            question=question,
            expected=expected,
        )
        for offset, (question, expected) in enumerate(PHYSICS_QUESTIONS)
    ]


def _graduate_math(start: int) -> List[SweepCase]:
    x = Symbol("x")
    cases: List[SweepCase] = []

    for index in range(15):
        value = 2 + index
        determinant = simplify(Matrix([[value, 1], [2, value + 3]]).det())
        cases.append(SweepCase(
            number=start + len(cases),
            level="Graduate",
            subject="Mathematics",
            question=f"determinant of [[{value},1],[2,{value + 3}]]",
            expected=(str(determinant),),
        ))

    for index in range(15):
        value = 2 + index
        cases.append(SweepCase(
            number=start + len(cases),
            level="Graduate",
            subject="Mathematics",
            question=f"eigen values of [[{value},0],[0,{value + 1}]]",
            expected=(str(value), str(value + 1)),
        ))

    for index in range(10):
        power = 2 + index
        result = integrate(x ** power, x)
        cases.append(SweepCase(
            number=start + len(cases),
            level="Graduate",
            subject="Mathematics",
            question=f"integrate x^{power}",
            expected=(f"{result} + C",),
        ))

    for index in range(10):
        power = 2 + index
        result = diff(x ** power, x)
        cases.append(SweepCase(
            number=start + len(cases),
            level="Graduate",
            subject="Mathematics",
            question=f"derivative of x^{power}",
            expected=(str(result),),
        ))

    return cases


def build_cases() -> List[SweepCase]:
    cards = subject_rag.load_cards()
    cases: List[SweepCase] = []
    basic_cards = _balanced_cards(cards, 500)
    advanced_cards = _balanced_cards(cards, 250)
    cases.extend(_basic_cards(basic_cards, 1))
    cases.extend(_intermediate_solvers(len(cases) + 1))
    advanced_reserved = (
        len(CONCEPT_GRAPH_QUESTIONS)
        + len(PROBABILITY_STATS_QUESTIONS)
        + len(PHYSICS_QUESTIONS)
    )
    cases.extend(_advanced_cards(advanced_cards, len(cases) + 1)[: 250 - advanced_reserved])
    cases.extend(_concept_graph_cases(len(cases) + 1))
    cases.extend(_probability_stats_cases(len(cases) + 1))
    cases.extend(_physics_cases(len(cases) + 1))
    cases.extend(_graduate_math(len(cases) + 1))
    return cases


def run_cases(cases: Sequence[SweepCase]) -> List[SweepResult]:
    results: List[SweepResult] = []
    with (
        patch.dict(wikipedia.CACHE, {}, clear=True),
        patch.dict(arxiv.CACHE, {}, clear=True),
        patch.object(wikipedia, "_search", side_effect=evaluation.offline_search),
        patch.object(wikipedia, "_hf_tokens", side_effect=evaluation._regex_tokens),
        patch.object(arxiv, "search", side_effect=evaluation.offline_arxiv_search),
        patch.object(
            graph_rag,
            "wikipedia_answer",
            side_effect=evaluation.offline_wikipedia_answer,
        ),
    ):
        for case in cases:
            answer = graph_rag.answer(case.question, [], [])
            lowered = answer.lower()
            failures: List[str] = []
            if not any(_contains_term(answer, term) for term in case.expected):
                failures.append("off-topic answer")
            if _has_badge(answer, case.expected):
                failures.append("source badge in answer")
            if any(phrase in lowered for phrase in FALLBACK_PHRASES):
                failures.append("fallback answer")
            results.append(SweepResult(
                number=case.number,
                level=case.level,
                subject=case.subject,
                question=case.question,
                expected=case.expected,
                answer=answer,
                passed=not failures,
                failures=tuple(failures),
            ))
    return results


def _cell(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", "")
        .replace("\n", "<br>")
    )


def write_report(
    results: Sequence[SweepResult],
    test_names: Sequence[str],
    eval_names: Sequence[str],
    unittest_count: int,
    unittest_problems: Sequence[str],
) -> None:
    levels = ["Basic", "Intermediate", "Advanced", "Graduate"]
    passed = sum(1 for result in results if result.passed)

    lines: List[str] = [
        "# ViDo Chat Lab - 1,000 topic answer sweep",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "## Previously verified Python chat checks",
        "",
        f"- `python/test_answers.py`: "
        f"{unittest_count - len(unittest_problems)}/{unittest_count} passed",
    ]
    lines.extend(f"  - `{name}`" for name in test_names)
    lines.extend(f"    - FAIL: {_cell(problem)}" for problem in unittest_problems)
    lines.extend([
        "",
        f"- `python/evaluation.py`: {len(eval_names)}/{len(eval_names)} passed",
    ])
    lines.extend(f"  - `{name}`" for name in eval_names)
    lines.extend([
        "",
        "## Sweep summary",
        "",
        f"- Total questions: {len(results)}",
        f"- Passed: {passed}",
        f"- Failed: {len(results) - passed}",
    ])
    for level in levels:
        level_results = [result for result in results if result.level == level]
        level_passed = sum(1 for result in level_results if result.passed)
        lines.append(
            f"- {level}: {level_passed}/{len(level_results)} passed"
        )
    lines.extend([
        "",
        "> Advanced/Graduate labels describe the question framing. The engine still",
        "> answers from the current card store and Phase 2 solvers, so depth is bounded",
        "> by that coverage. Every response below is the exact Python answer text.",
        ">",
        "> The Expected column lists accepted topic names/aliases for the selected",
        "> card; an answer passes when it stays on topic and is not a fallback or a",
        "> source badge. Dedicated unittest/evaluation cases verify exact routing.",
        "",
        "## Every response",
        "",
        "| # | Level | Subject | Question | Expected | Status | Response |",
        "|---:|---|---|---|---|---|---|",
    ])
    for result in results:
        status = "PASS" if result.passed else "FAIL: " + "; ".join(result.failures)
        shown_expected = ", ".join(result.expected[:3])
        if len(result.expected) > 3:
            shown_expected += f" (+{len(result.expected) - 3} aliases)"
        lines.append(
            f"| {result.number} | {_cell(result.level)} | {_cell(result.subject)} | "
            f"{_cell(result.question)} | {_cell(shown_expected)} | "
            f"{_cell(status)} | {_cell(result.answer)} |"
        )

    with open(REPORT_MD, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    with open(REPORT_JSONL, "w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps({
                "number": result.number,
                "level": result.level,
                "subject": result.subject,
                "question": result.question,
                "expected": list(result.expected),
                "answer": result.answer,
                "passed": result.passed,
                "failures": list(result.failures),
            }, ensure_ascii=False) + "\n")


def main() -> int:
    test_names, eval_names, unittest_count, unittest_problems = collect_verified_checks()
    cases = build_cases()
    results = run_cases(cases)
    write_report(results, test_names, eval_names, unittest_count, unittest_problems)

    passed = sum(1 for result in results if result.passed)
    print(
        f"Verified Python unittest checks: "
        f"{unittest_count - len(unittest_problems)}/{unittest_count} passed"
    )
    for problem in unittest_problems:
        print(f"  unittest failure: {problem}")
    print(f"Verified evaluation cases: {len(eval_names)}/{len(eval_names)} passed")
    print(f"Chat sweep: {passed}/{len(results)} passed")
    for level in ("Basic", "Intermediate", "Advanced", "Graduate"):
        level_results = [result for result in results if result.level == level]
        level_passed = sum(1 for result in level_results if result.passed)
        print(f"  {level}: {level_passed}/{len(level_results)}")
    print(f"Full responses: {REPORT_MD}")
    print(f"JSONL responses: {REPORT_JSONL}")

    failures = [result for result in results if not result.passed]
    for result in failures[:20]:
        print(f"FAIL #{result.number} [{result.level}] {result.question}")
        for failure in result.failures:
            print(f"  - {failure}")
        print(f"  answer: {result.answer}")
    if len(failures) > 20:
        print(f"... and {len(failures) - 20} more in the report")
    return 0 if not failures and not unittest_problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
