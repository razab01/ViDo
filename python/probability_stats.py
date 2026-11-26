"""Probability and statistics knowledge plus computation for ViDo.

This module is the broad companion to the smaller coin/dice helpers in
``vido_math``. It combines three ideas:

* a curated glossary grounded in the shared subject-card corpus,
* deterministic numerical solvers for event algebra, distributions, sampling,
  counting and inference, and
* short RAG-grounded summaries drawn from the same card store.

Answers stay badge-free: supporting books, encyclopedia articles and papers
remain internal evidence, never visible source labels.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import subject_rag


OVERVIEW = (
    "Probability measures how likely an event is, from 0 (impossible) to 1 (certain). "
    "It starts with a sample space of all outcomes; when outcomes are equally likely, "
    "P(A) = favourable/total. Core rules are P(not A) = 1 - P(A), "
    "P(A or B) = P(A) + P(B) - P(A and B), P(A and B) = P(A)P(B) for independent events, "
    "and P(A given B) = P(A and B)/P(B). A random variable assigns numbers to outcomes, "
    "and its distribution gives their probabilities or density; expected value and variance "
    "describe its centre and spread. Statistics uses probability as a model for data: it "
    "summarises samples with the mean and variance, then uses estimation, confidence intervals "
    "and hypothesis tests for inference about a population. The law of large numbers and the "
    "central limit theorem are the bridge between the two."
)


CONCEPT_TITLES: Dict[str, str] = {
    "sample_space": "Sample space",
    "axioms": "Axioms of probability",
    "mutually_exclusive": "Mutually exclusive events",
    "independent": "Independent events",
    "conditional": "Conditional probability",
    "bayes": "Bayes theorem",
    "random_variable": "Random variable",
    "pmf": "Probability mass function",
    "pdf": "Probability density function",
    "cdf": "Cumulative distribution function",
    "expectation": "Expected value",
    "variance": "Variance",
    "standard_deviation": "Standard deviation",
    "covariance": "Covariance",
    "correlation": "Correlation",
    "chebyshev": "Chebyshev inequality",
    "lln": "Law of large numbers",
    "clt": "Central limit theorem",
    "bernoulli": "Bernoulli distribution",
    "binomial": "Binomial distribution",
    "poisson": "Poisson distribution",
    "geometric": "Geometric distribution",
    "exponential": "Exponential distribution",
    "hypergeometric": "Hypergeometric distribution",
    "uniform": "Uniform distribution",
    "hypothesis_testing": "Hypothesis testing",
    "p_value": "P-value",
    "confidence_interval": "Confidence interval",
    "type_i_error": "Type I error",
    "type_ii_error": "Type II error",
    "power": "Statistical power",
    "regression": "Regression",
    "sampling_distribution": "Sampling distribution",
    "probability_distribution": "Probability distribution",
}


CONCEPTS: Dict[str, str] = {
    "sample_space": (
        "The sample space is the set of all possible outcomes of a random experiment. "
        "For one coin toss it is {head, tail}; for two dice it contains all 36 ordered pairs. "
        "Every event is a subset of the sample space."
    ),
    "axioms": (
        "The probability axioms say that probabilities are non-negative, P(Ω) = 1, and for "
        "mutually exclusive events P(A ∪ B) = P(A) + P(B) (countable additivity). "
        "Together they imply every probability lies between 0 and 1."
    ),
    "mutually_exclusive": (
        "Mutually exclusive events cannot happen at the same time, so P(A and B) = 0 and "
        "P(A or B) = P(A) + P(B). They are different from independent events, where one event "
        "does not change the other's probability."
    ),
    "independent": (
        "Two events are independent when one does not affect the probability of the other: "
        "P(A given B) = P(A), which is equivalent to P(A and B) = P(A)P(B). Independence is "
        "not the same as being mutually exclusive."
    ),
    "conditional": (
        "Conditional probability is the probability of A given that B has happened: "
        "P(A|B) = P(A and B)/P(B) when P(B) > 0. It narrows the sample space to the outcomes "
        "where B is true."
    ),
    "bayes": (
        "Bayes theorem updates a prior belief using new evidence: "
        "P(A|B) = P(B|A)P(A)/P(B), and with a binary complement "
        "P(A|B) = P(B|A)P(A) / [P(B|A)P(A) + P(B|not A)P(not A)]. "
        "P(A) is the prior, P(B|A) the likelihood, and P(A|B) the posterior."
    ),
    "random_variable": (
        "A random variable maps outcomes of a random experiment to numbers. A discrete random "
        "variable takes countable values; a continuous one takes values in intervals. Its "
        "distribution tells you the probability of each value or range."
    ),
    "pmf": (
        "A probability mass function gives P(X = k) for a discrete random variable. "
        "Its values are non-negative and sum to 1 over all possible k."
    ),
    "pdf": (
        "A probability density function describes a continuous random variable. The density "
        "itself is not a probability; probability is area under the curve, "
        "P(a ≤ X ≤ b) = ∫ from a to b of f(x)dx, and the total area is 1."
    ),
    "cdf": (
        "The cumulative distribution function is F(x) = P(X ≤ x): the probability that X is "
        "less than or equal to x. It is non-decreasing, right-continuous, and runs from 0 to 1. "
        "For a continuous variable the density is the derivative of the CDF; for a discrete "
        "variable the CDF adds PMF values."
    ),
    "expectation": (
        "Expected value is the long-run weighted average of a random variable: "
        "E[X] = Σ x·P(X=x) for discrete outcomes, or ∫x·f(x)dx for a continuous density. "
        "It is linear: E[aX+b] = aE[X]+b."
    ),
    "variance": (
        "Variance measures average squared spread around the mean: Var(X) = E[(X-μ)²] = "
        "E[X²] - (E[X])². Its square root is the standard deviation, which is in the "
        "original units."
    ),
    "standard_deviation": (
        "Standard deviation is the square root of variance. It describes the typical distance "
        "of values from the mean in the same units as the data; smaller means tighter clustering."
    ),
    "covariance": (
        "Covariance measures how two random variables vary together: Cov(X,Y) = "
        "E[(X-E[X])(Y-E[Y])]. Positive means they tend to rise together, negative means one "
        "rises as the other falls, and zero means no linear co-movement."
    ),
    "correlation": (
        "Correlation measures the strength and direction of a linear relationship. The "
        "correlation coefficient r lies between -1 and 1: -1 is perfect negative, 0 is no "
        "linear relationship, and +1 is perfect positive. Correlation does not imply causation."
    ),
    "chebyshev": (
        "Chebyshev inequality says that for any distribution with finite variance, at least "
        "1 - 1/k² of values lie within k standard deviations of the mean. It is weaker than "
        "the normal rule but does not assume normality."
    ),
    "lln": (
        "The law of large numbers says that as you repeat an experiment many times, the sample "
        "average converges to the expected value. It explains why long-run relative frequency "
        "stabilises near the true probability."
    ),
    "clt": (
        "The central limit theorem says that the sample mean of many independent, "
        "similarly distributed observations is approximately normal once the sample is large "
        "enough, whatever the original distribution. That is why normal approximations and "
        "z-tests work so broadly."
    ),
    "bernoulli": (
        "A Bernoulli distribution models one trial with two outcomes: success (1) and failure "
        "(0). If P(success) = p, then P(X=1)=p, P(X=0)=1-p, E[X]=p and Var(X)=p(1-p)."
    ),
    "binomial": (
        "A binomial distribution counts successes in n independent trials with the same "
        "success probability p: P(X=k)=C(n,k)·p^k·(1-p)^(n-k), E[X]=np and Var(X)=np(1-p)."
    ),
    "poisson": (
        "A Poisson distribution counts events in a fixed interval of time or space when they "
        "occur independently at a constant average rate λ: P(X=k)=e^(-λ)·λ^k/k!, and both "
        "E[X] and Var(X) equal λ."
    ),
    "geometric": (
        "A geometric distribution counts the number of trials until the first success in "
        "independent trials with success probability p: P(X=k)=(1-p)^(k-1)·p, E[X]=1/p and "
        "Var(X)=(1-p)/p²."
    ),
    "exponential": (
        "An exponential distribution models waiting time between events in a Poisson process "
        "with rate λ: f(x)=λe^(-λx), P(X>x)=e^(-λx), E[X]=1/λ and Var(X)=1/λ²."
    ),
    "hypergeometric": (
        "A hypergeometric distribution counts successes when sampling without replacement from "
        "a finite population. With K successes in N items and n draws, "
        "P(X=k)=C(K,k)·C(N-K,n-k)/C(N,n). Unlike the binomial case, trials are not independent."
    ),
    "uniform": (
        "A uniform distribution gives equally likely outcomes. For a finite set each outcome "
        "has probability 1/n; for a continuous interval [a,b] the density is constant at "
        "1/(b-a), so intervals of equal length have equal probability."
    ),
    "hypothesis_testing": (
        "Hypothesis testing starts with a null hypothesis and an alternative. You choose a "
        "significance level, compute a test statistic and p-value, then reject the null only "
        "if the evidence is strong enough. It controls error rates rather than proving a claim."
    ),
    "p_value": (
        "A p-value is the probability of data at least as extreme as what was observed, "
        "assuming the null hypothesis is true. A small p-value argues against the null; it is "
        "not the probability that the null is true."
    ),
    "confidence_interval": (
        "A confidence interval is a range of plausible values for a population parameter. "
        "A 95% interval is built so that the method captures the true parameter in about 95% "
        "of repeated samples; it does not mean there is a 95% chance the fixed parameter "
        "lies in this one computed interval."
    ),
    "type_i_error": (
        "A Type I error is rejecting a true null hypothesis, a false positive. Its probability "
        "is the significance level α."
    ),
    "type_ii_error": (
        "A Type II error is failing to reject a false null hypothesis, a false negative. Its "
        "probability is β; statistical power is 1 - β."
    ),
    "power": (
        "Statistical power is the probability that a test rejects a false null hypothesis, "
        "so it is the chance of detecting a real effect. Power rises with sample size and "
        "effect size and falls with stricter significance thresholds."
    ),
    "regression": (
        "Regression models the relationship between a response or dependent variable and one "
        "or more predictors so you can describe and predict outcomes. Linear regression fits "
        "a straight line by least squares; residuals and R² show how well it fits."
    ),
    "sampling_distribution": (
        "A sampling distribution is the distribution of a sample statistic, such as the sample "
        "mean, over repeated samples of the same size. Its spread is the standard error, and "
        "the central limit theorem often makes it approximately normal."
    ),
    "probability_distribution": (
        "A probability distribution assigns probabilities to the possible outcomes of a random "
        "variable. Discrete variables use a probability mass function P(X=x); continuous ones "
        "use a density whose areas give probabilities."
    ),
}


CONTRASTS: Dict[Tuple[str, str], str] = {
    ("poisson", "binomial"): (
        "Binomial counts successes in a fixed number n of trials; Poisson counts events in a "
        "fixed interval when only the average rate λ is fixed. Poisson is the rare-event limit "
        "of binomial when n is large and p is small."
    ),
    ("expectation", "variance"): (
        "Expectation is the centre (long-run average), while variance is the spread around that "
        "centre. Variance is E[(X-E[X])²], not a second kind of average."
    ),
    ("conditional", "independent"): (
        "Conditional probability updates a probability after learning B; independence is the "
        "special case where that update changes nothing."
    ),
}

CONCEPT_ALIASES: Dict[str, str] = {
    "sample space": "sample_space",
    "outcomes": "sample_space",
    "axioms of probability": "axioms",
    "probability axioms": "axioms",
    "kolmogorov axioms": "axioms",
    "mutually exclusive": "mutually_exclusive",
    "mutually exclusive events": "mutually_exclusive",
    "independent": "independent",
    "independent events": "independent",
    "independence": "independent",
    "conditional probability": "conditional",
    "conditional": "conditional",
    "bayes theorem": "bayes",
    "bayes rule": "bayes",
    "bayes' theorem": "bayes",
    "posterior": "bayes",
    "random variable": "random_variable",
    "probability mass function": "pmf",
    "pmf": "pmf",
    "probability density function": "pdf",
    "pdf": "pdf",
    "cumulative distribution function": "cdf",
    "cdf": "cdf",
    "expected value": "expectation",
    "expectation": "expectation",
    "mean": "expectation",
    "variance": "variance",
    "standard deviation": "standard_deviation",
    "covariance": "covariance",
    "correlation": "correlation",
    "correlation coefficient": "correlation",
    "chebyshev inequality": "chebyshev",
    "chebyshev's inequality": "chebyshev",
    "law of large numbers": "lln",
    "central limit theorem": "clt",
    "bernoulli distribution": "bernoulli",
    "binomial distribution": "binomial",
    "poisson distribution": "poisson",
    "geometric distribution": "geometric",
    "exponential distribution": "exponential",
    "hypergeometric distribution": "hypergeometric",
    "uniform distribution": "uniform",
    "hypothesis testing": "hypothesis_testing",
    "hypothesis test": "hypothesis_testing",
    "null hypothesis": "hypothesis_testing",
    "p-value": "p_value",
    "p value": "p_value",
    "confidence interval": "confidence_interval",
    "type i error": "type_i_error",
    "type ii error": "type_ii_error",
    "statistical power": "power",
    "power of a test": "power",
    "regression": "regression",
    "linear regression": "regression",
    "sampling distribution": "sampling_distribution",
    "probability distribution": "probability_distribution",
}

SHORT_ALIASES: Dict[str, str] = {
    "poisson": "poisson",
    "binomial": "binomial",
    "bernoulli": "bernoulli",
    "geometric": "geometric",
    "exponential": "exponential",
    "uniform": "uniform",
    "hypergeometric": "hypergeometric",
    "expectation": "expectation",
    "variance": "variance",
    "conditional": "conditional",
    "independence": "independent",
    "independent": "independent",
    "bayes": "bayes",
    "correlation": "correlation",
    "covariance": "covariance",
}

_TYPOS = {
    "theorm": "theorem",
    "standrad": "standard",
    "diced": "sided",
    "probablity": "probability",
    "poission": "poisson",
    "condtional": "conditional",
    "independant": "independent",
    "distribtion": "distribution",
    "hyothesis": "hypothesis",
    "confidence inteval": "confidence interval",
}

_NUMBER_WORDS = {
    "zero": 0, "a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "twice": 2,
    "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "twenty-one": 21,
    "twenty-two": 22, "twenty-three": 23, "twenty-four": 24,
    "twenty-five": 25, "twenty-six": 26, "twenty-seven": 27,
    "twenty-eight": 28, "twenty-nine": 29, "thirty": 30,
}

_TOPIC_GATE = re.compile(
    r"\b(?:probability|probabilities|chance|likelihood|odds|event|events|random\s+variable|"
    r"distribution|expected\s+value|expectation|variance|standard\s+deviation|standard\s+error|"
    r"covariance|correlation|regression|bayes|conditional|independence|independent|"
    r"mutually\s+exclusive|sample\s+space|axioms|sampling|hypothesis|p-value|confidence\s+interval|"
    r"type\s+i\s+error|type\s+ii\s+error|statistical\s+power|"
    r"central\s+limit|law\s+of\s+large\s+numbers|chebyshev|bernoulli|binomial|poisson|geometric|"
    r"exponential|hypergeometric|uniform|normal|normally|z-score|z\s+statistic|trial|trials|success|successes|"
    r"dice|die|deck|card|urn|permutation|permutations|combinations?|choose|confidence|p\s*\(|"
    r"coin|heads?|tails?|toss(?:es|ed|ing)?|flip(?:s|ped|ping)?|expected\s+number|flagged|spam)",
    re.I,
)


def _normalize(question: str) -> str:
    value = str(question or "").lower()
    value = re.sub(r"\s+", " ", value).strip(" .?!")
    for wrong, right in _TYPOS.items():
        value = re.sub(rf"\b{re.escape(wrong)}\b", right, value)
    return value


def _fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return f"{round(value, 4):g}"


def _fraction(value: float, max_denominator: int = 1000) -> Optional[str]:
    frac = Fraction(value).limit_denominator(max_denominator)
    if abs(float(frac) - value) < 1e-9 and frac.denominator > 1:
        return f"{frac.numerator}/{frac.denominator}"
    return None


def _prob(value: float, explanation: str, fraction: Optional[str] = None) -> str:
    fraction = fraction or _fraction(value)
    percent = f"{_fmt(value * 100)}%"
    rendered = _fmt(value)
    rendered += f" ({fraction}, or {percent})" if fraction else f" ({percent})"
    return f"{rendered}. {explanation}"


def _labeled_probability(question: str, pattern: str) -> Optional[float]:
    match = re.search(
        pattern + r"\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?\s*%?)",
        question,
        re.I,
    )
    if not match:
        return None
    raw = match.group(1).strip()
    value = float(raw[:-1]) if raw.endswith("%") else float(raw)
    return value / 100 if raw.endswith("%") else value


def _percent(raw: str) -> float:
    value = raw.strip()
    return float(value[:-1]) / 100 if value.endswith("%") else float(value)


def _percent_match(question: str, pattern: str) -> Optional[float]:
    """Capture the percentage immediately before or after a phrase."""
    match = re.search(pattern + r"\s*(\d+(?:\.\d+)?\s*%?)", question, re.I)
    if not match:
        match = re.search(r"(\d+(?:\.\d+)?\s*%?)\s*" + pattern, question, re.I)
    return _percent(match.group(1)) if match else None


def _rag_support(key: str) -> str:
    """Append a short grounded note from the shared card store when available."""
    alias = CONCEPT_TITLES.get(key, key).lower()
    try:
        ranked = subject_rag.rank_cards(alias)
    except Exception:
        return ""
    for _score, _matched, card in ranked:
        if card.subject_key not in {
            "math", "math_branches", "statistics_branches", "probability_statistics"
        }:
            continue
        extras: List[str] = []
        if card.formula and not card.formula.lower().startswith("branch:"):
            extras.append(f"Quick form: {card.formula}")
        if card.points and card.points != card.formula:
            extras.append(f"Remember: {card.points}")
        if extras:
            return " " + " ".join(extras)
        break
    return ""


def _concept_text(key: str) -> str:
    answer = CONCEPTS[key]
    if key == "bayes" and "textbook-style example" in _CURRENT_QUESTION:
        answer += (
            " Example: with 1% prevalence, 99% sensitivity and a 5% false-positive rate, "
            "P(disease|positive) = 0.99·0.01 / (0.99·0.01 + 0.05·0.99) ≈ 0.1667."
        )
    if key == "lln" and "feller" in _CURRENT_QUESTION:
        answer += (
            " In Feller's classic treatment, the point is exactly this long-run average: "
            "random fluctuation averages out as the number of trials grows."
        )
    return answer + _rag_support(key)


_CURRENT_QUESTION = ""


def _matched_concepts(question: str, short: bool = False) -> List[str]:
    aliases = SHORT_ALIASES if short else CONCEPT_ALIASES
    matches: List[Tuple[int, int, str]] = []
    for alias, key in aliases.items():
        for match in re.finditer(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", question):
            matches.append((match.start(), -len(alias), key))
    seen = set()
    ordered: List[str] = []
    for start, _negative_length, key in sorted(matches):
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _concept_answer(question: str) -> Optional[str]:
    global _CURRENT_QUESTION
    low = _normalize(question)
    _CURRENT_QUESTION = low
    if not _TOPIC_GATE.search(low):
        return None

    if (
        "probability and statistics" in low
        or "statistics and probability" in low
        or "how are probability and statistics related" in low
    ):
        return OVERVIEW

    compare_cue = bool(
        re.search(r"\b(?:compare|versus|vs\.?|difference between|different from)\b", low)
        or " and " in low and low.startswith(("compare ", "tell me the difference between "))
    )
    if compare_cue:
        keys = _matched_concepts(low, short=True)
        if len(keys) >= 2:
            first, second = keys[0], keys[1]
            contrast = CONTRASTS.get(
                (first, second), CONTRASTS.get((second, first))
            )
            if not contrast:
                contrast = (
                    f"{CONCEPT_TITLES[first]} focuses on one part of probability and statistics, "
                    f"while {CONCEPT_TITLES[second]} answers a different question."
                )
            return (
                f"Compare {CONCEPT_TITLES[first]} and {CONCEPT_TITLES[second]}.\n\n"
                f"{_concept_text(first)}\n\n{_concept_text(second)}\n\n"
                f"Key difference: {contrast}"
            )

    definition_cue = bool(
        re.match(
            r"^(?:what\s+(?:is|are|does)|define|explain|describe|meaning\s+of|tell\s+me\s+about|why\s+does|how\s+are)\b",
            low,
        )
    )
    if definition_cue:
        keys = _matched_concepts(low)
        if keys:
            return _concept_text(keys[0])
    return None


def _event_algebra(question: str) -> Optional[str]:
    if not re.search(r"\bP\s*\(|\bprobability\b|\bchance\b", question, re.I):
        return None

    pa = _labeled_probability(question, r"p\s*\(\s*a\s*\)")
    pb = _labeled_probability(question, r"p\s*\(\s*b\s*\)")
    pab = _labeled_probability(
        question,
        r"p\s*\(\s*(?:a\s*(?:∩|\band\s*)\s*b|b\s*(?:∩|\band\s*)\s*a)\s*\)",
    )

    if pa is not None and re.search(r"\b(?:not\s+a|complement\s+of\s+a|a\s+complement)\b", question, re.I):
        value = 1 - pa
        return _prob(value, f"P(not A) = 1 - P(A) = 1 - {_fmt(pa)}.")

    if pab is not None and pb is not None and re.search(r"\bgiven\b|\bconditional\b", question, re.I):
        if pb <= 0:
            return None
        value = pab / pb
        return _prob(
            value,
            f"P(A given B) = P(A and B)/P(B) = {_fmt(pab)}/{_fmt(pb)} = {_fmt(value)}.",
        )

    if (
        pa is not None
        and pb is not None
        and pab is not None
        and re.search(r"\b(?:are|is)\b[^?]*\bindependent\b", question, re.I)
    ):
        expected = pa * pb
        independent = abs(pab - expected) < 1e-9
        answer = "Yes" if independent else "No"
        relation = "=" if independent else "≠"
        return (
            f"{answer} — P(A and B) = {_fmt(pab)} and P(A)·P(B) = "
            f"{_fmt(pa)}·{_fmt(pb)} = {_fmt(expected)}, so {_fmt(pab)} "
            f"{relation} {_fmt(expected)}."
        )

    union_cue = bool(
        re.search(r"\bunion\b|\bor\b|\beither\b", question, re.I)
        or "∪" in question
    )
    if pa is not None and pb is not None and pab is not None and union_cue:
        value = pa + pb - pab
        return _prob(
            value,
            f"P(A or B) = P(A) + P(B) - P(A and B) = {_fmt(pa)} + {_fmt(pb)} - {_fmt(pab)} = {_fmt(value)}.",
        )

    if (
        pa is not None
        and pb is not None
        and pab is None
        and union_cue
        and re.search(r"\bindependent\b", question, re.I)
    ):
        overlap = pa * pb
        value = pa + pb - overlap
        return _prob(
            value,
            f"For independent events, P(A or B) = P(A) + P(B) - P(A)P(B) = "
            f"{_fmt(pa)} + {_fmt(pb)} - {_fmt(pa)}·{_fmt(pb)} = {_fmt(value)}.",
        )

    if pa is not None and pb is not None and re.search(r"\bmutually exclusive\b", question, re.I):
        value = pa + pb
        return _prob(value, f"Mutually exclusive means P(A and B)=0, so P(A or B) = {_fmt(pa)} + {_fmt(pb)} = {_fmt(value)}.")

    if pa is not None and pb is not None and re.search(r"\bindependent\b", question, re.I):
        value = pa * pb
        return _prob(
            value,
            f"Independent events satisfy P(A and B) = P(A)·P(B) = {_fmt(pa)}·{_fmt(pb)} = {_fmt(value)}.",
        )
    return None


def _bayes_medical(question: str) -> Optional[str]:
    if re.search(r"\bspam\b|\bflagged\b", question, re.I):
        prevalence = re.search(
            r"(\d+(?:\.\d+)?)\s*%\s*of\s+(?:mail|messages?|emails?)\s+is\s+spam",
            question,
            re.I,
        )
        sensitivity = re.search(
            r"flags?\s+(\d+(?:\.\d+)?)\s*%\s*of\s+spam", question, re.I
        )
        false_positive = re.search(
            r"(\d+(?:\.\d+)?)\s*%\s*of\s+legitimate", question, re.I
        )
        if prevalence and sensitivity and false_positive:
            base = float(prevalence.group(1)) / 100
            hit = float(sensitivity.group(1)) / 100
            false_alarm = float(false_positive.group(1)) / 100
            denominator = hit * base + false_alarm * (1 - base)
            if denominator <= 0:
                return None
            value = hit * base / denominator
            return _prob(
                value,
                f"By Bayes theorem, P(spam|flagged) = "
                f"({_fmt(hit)}·{_fmt(base)}) / "
                f"({_fmt(hit)}·{_fmt(base)} + {_fmt(false_alarm)}·{_fmt(1 - base)}) "
                f"= {_fmt(value)}.",
            )
        return None
    if not re.search(r"\btest\b|\bscreen\b", question, re.I):
        return None
    if not re.search(r"\bdisease\b|\bcondition\b|\billness\b", question, re.I):
        return None
    prevalence = _labeled_probability(
        question, r"(?:affects|prevalence(?:\s+of)?(?:\s+is)?)"
    )
    sensitivity = _labeled_probability(
        question, r"(\d+(?:\.\d+)?\s*%\s*)?(?=sensitive|sensitivity)"
    )
    false_positive = _labeled_probability(
        question, r"(\d+(?:\.\d+)?\s*%\s*)?(?=false\s+positive)"
    )
    if prevalence is None or sensitivity is None or false_positive is None:
        # The lookahead helpers above capture the number before the phrase, so
        # fall back to direct searches when needed.
        prev_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s+people|affects\s+(\d+(?:\.\d+)?)\s*%", question, re.I)
        sens_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*sensitive|sensitivity\s*(?:is|=)?\s*(\d+(?:\.\d+)?)\s*%", question, re.I)
        fp_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*false\s+positive|false\s+positive\s+rate\s*(?:is|=)?\s*(\d+(?:\.\d+)?)\s*%", question, re.I)
        if not (prev_match and sens_match and fp_match):
            return None
        def first(match):
            return next(group for group in match.groups() if group is not None)
        prevalence = float(first(prev_match)) / 100
        sensitivity = float(first(sens_match)) / 100
        false_positive = float(first(fp_match)) / 100

    true_positive = sensitivity * prevalence
    false_positive_mass = false_positive * (1 - prevalence)
    denominator = true_positive + false_positive_mass
    if denominator <= 0:
        return None
    posterior = true_positive / denominator
    return _prob(
        posterior,
        "By Bayes theorem, P(disease|positive) = "
        f"({_fmt(sensitivity)}·{_fmt(prevalence)}) / "
        f"({_fmt(sensitivity)}·{_fmt(prevalence)} + {_fmt(false_positive)}·{_fmt(1 - prevalence)}) "
        f"= {_fmt(posterior)}.",
    )


def _count_from_text(raw: str) -> Optional[int]:
    value = raw.strip().lower()
    if value.isdigit():
        return int(value)
    return _NUMBER_WORDS.get(value)


_NUMBER_TOKEN = r"\d+|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))


def _first_group(match: Optional[re.Match]) -> Optional[str]:
    if not match:
        return None
    return next((group for group in match.groups() if group is not None), None)


def _count_match(question: str, pattern: str) -> Optional[int]:
    raw = _first_group(re.search(pattern, question, re.I))
    return _count_from_text(raw) if raw is not None else None


def _sides(question: str) -> Optional[int]:
    sides = _count_match(question, rf"\b({_NUMBER_TOKEN})\s*(?:-|–|—|\s)?sided\b")
    return sides if sides is not None else 6


def _hypergeom_pmf(population: int, successes: int, draws: int, observed: int) -> float:
    if not (0 <= observed <= successes and 0 <= draws - observed <= population - successes):
        return 0.0
    return (
        math.comb(successes, observed)
        * math.comb(population - successes, draws - observed)
        / math.comb(population, draws)
    )


def _head_count_spec(chunk: str, total: int) -> Optional[Tuple[str, int]]:
    """Parse a head/tail count phrase into an event about the number of heads."""
    patterns = (
        (rf"\b(?:exactly|precisely|just|only)\s+({_NUMBER_TOKEN})\s+(heads?|tails?)\b", "exact"),
        (rf"\bat\s+least\s+({_NUMBER_TOKEN})\s+(heads?|tails?)\b", "at_least"),
        (rf"\bat\s+most\s+({_NUMBER_TOKEN})\s+(heads?|tails?)\b", "at_most"),
        (r"\b(?:no|none|zero)\s+(heads?|tails?)\b", "zero"),
        (r"\ball\s+(heads?|tails?)\b", "all"),
        (rf"\b({_NUMBER_TOKEN})\s+(heads?|tails?)\b", "exact"),
    )
    for pattern, mode in patterns:
        match = re.search(pattern, chunk, re.I)
        if not match:
            continue
        if mode in {"zero", "all"}:
            raw_outcome = match.group(1)
            k = 0 if mode == "zero" else total
            mode = "exact"
        else:
            k = _count_from_text(match.group(1))
            raw_outcome = match.group(2)
        if k is None:
            return None
        if raw_outcome.lower().startswith("h"):
            return mode, k
        if mode == "exact":
            return "exact", total - k
        if mode == "at_least":
            return "at_most", total - k
        if mode == "at_most":
            return "at_least", total - k
        return "exact", total
    return None


def _count_holds(count: int, mode: str, threshold: int) -> bool:
    if mode == "exact":
        return count == threshold
    if mode == "at_least":
        return count >= threshold
    return count <= threshold


def _count_spec_label(mode: str, threshold: int) -> str:
    plural = "s" if threshold != 1 else ""
    if mode == "exact":
        return f"exactly {threshold} head{plural}"
    if mode == "at_least":
        return f"at least {threshold} head{plural}"
    return f"at most {threshold} head{plural}"


def _conditional_probability(question: str) -> Optional[str]:
    cue = re.search(r"\b(?:given that|given|conditional on|knowing that)\b", question, re.I)
    if not cue:
        return None

    card_terms = r"ace|king|queen|jack|spade|heart|club|diamond"
    card_condition = re.search(
        rf"\bgiven\s+that\s+it\s+is\s+an?\s+({card_terms})\b", question, re.I
    )
    card_target = re.search(
        rf"\bprobability\s+that\s+it\s+is\s+an?\s+({card_terms})\b", question, re.I
    )
    if card_condition and card_target:
        condition_name = card_condition.group(1).lower()
        target_name = card_target.group(1).lower()
        ranks = {"ace", "king", "queen", "jack"}
        suits = {"spade", "heart", "club", "diamond"}
        qualifies = (
            condition_name in ranks and target_name in suits
        ) or (
            condition_name in suits and target_name in ranks
        )
        if qualifies:
            total = 4 if condition_name in ranks else 13
            return _prob(
                1 / total,
                f"Given the card is {condition_name}, the sample space has {total} "
                f"cards. Exactly 1 of them is {target_name}, so the conditional "
                f"probability is 1/{total}.",
                f"1/{total}",
            )

    shown_face = _count_match(
        question,
        rf"\bgiven\s+that\s+at\s+least\s+one\s+(?:die|dice)\s+(?:shows?|is)\s+"
        rf"(?:a\s+)?({_NUMBER_TOKEN})\b",
    )
    dice_target = _count_match(
        question, rf"\bsum\s+(?:is|of|equals?|=)\s*({_NUMBER_TOKEN})\b"
    )
    sides = _sides(question)
    if shown_face is not None and dice_target is not None and sides:
        pairs = [(a, b) for a in range(1, sides + 1) for b in range(1, sides + 1)]
        conditioned = [pair for pair in pairs if shown_face in pair]
        favourable = [pair for pair in conditioned if sum(pair) == dice_target]
        if conditioned:
            reduced = math.gcd(len(favourable), len(conditioned))
            fraction = (
                f"{len(favourable) // reduced}/{len(conditioned) // reduced}"
                if favourable
                else None
            )
            return _prob(
                len(favourable) / len(conditioned),
                f"Given at least one die shows {shown_face}, there are "
                f"{len(conditioned)} of {sides}² ordered outcomes left. "
                f"{len(favourable)} of them have sum {dice_target}, so the "
                f"conditional probability is {len(favourable)}/{len(conditioned)}.",
                fraction,
            )

    first_die = _count_match(
        question, rf"\bfirst\s+die\s+(?:shows|is|equals)\s+({_NUMBER_TOKEN})\b"
    )
    if first_die is not None:
        target = _count_match(
            question, rf"\bsum\s+(?:is|of|equals?|=)\s*({_NUMBER_TOKEN})\b"
        )
        sides = _sides(question)
        if target is not None and sides and 1 <= first_die <= sides:
            favourable = sum(
                1
                for face in range(1, sides + 1)
                if first_die + face == target
            )
            reduced = math.gcd(favourable, sides)
            fraction = f"{favourable // reduced}/{sides // reduced}" if favourable else None
            return _prob(
                favourable / sides,
                f"With the first die fixed at {first_die}, only the second die is "
                f"random: {sides} equally likely faces. {favourable} of them make the "
                f"sum {target}, so P(sum = {target} given first die = {first_die}) "
                f"= {favourable}/{sides}; the probability is {_fmt(favourable / sides)}.",
                fraction,
            )

    tosses = _count_match(
        question,
        rf"\b({_NUMBER_TOKEN})\s+(?:fair\s+)?(?:coin\s+)?(?:times|tosses|flips|time|toss|flip)\b",
    )
    if tosses is None and re.search(r"\btwice\b", question, re.I):
        tosses = 2
    if tosses is None or tosses <= 0:
        return None

    prefix = question[:cue.start()]
    remainder = question[cue.end():]
    pieces = re.split(r"[,;]", remainder, maxsplit=1)
    condition_chunk = pieces[0]
    target_chunk = prefix + (" " + pieces[1] if len(pieces) > 1 else "")
    condition = _head_count_spec(condition_chunk, tosses)
    target = _head_count_spec(target_chunk, tosses)

    first_flip = re.search(
        rf"\bfirst\s+(?:flip|toss)\s+is\s+(heads?|tails?)\b", condition_chunk, re.I
    )
    if first_flip and target:
        fixed_heads = 1 if first_flip.group(1).lower().startswith("h") else 0
        remaining = tosses - 1
        qualifying = [
            heads
            for heads in range(remaining + 1)
            if _count_holds(heads + fixed_heads, target[0], target[1])
        ]
        numerator = sum(math.comb(remaining, heads) for heads in qualifying)
        denominator = 2 ** remaining
        reduced = math.gcd(numerator, denominator)
        fraction = f"{numerator // reduced}/{denominator // reduced}" if numerator else None
        fixed_name = "heads" if fixed_heads else "tails"
        return _prob(
            numerator / denominator,
            f"With the first flip fixed at {fixed_name}, only {remaining} flips "
            f"remain: {denominator} equally likely outcomes. {numerator} of them "
            f"give {_count_spec_label(*target)} in total, so the conditional "
            f"probability is {numerator}/{denominator}.",
            fraction,
        )

    if not condition or not target:
        return None

    conditioned = [
        heads for heads in range(tosses + 1)
        if _count_holds(heads, condition[0], condition[1])
    ]
    favourable = [
        heads for heads in conditioned
        if _count_holds(heads, target[0], target[1])
    ]
    denominator = sum(math.comb(tosses, heads) for heads in conditioned)
    if denominator <= 0:
        return None
    numerator = sum(math.comb(tosses, heads) for heads in favourable)
    reduced = math.gcd(numerator, denominator)
    fraction = f"{numerator // reduced}/{denominator // reduced}" if numerator else None
    return _prob(
        numerator / denominator,
        f"Conditioning on {_count_spec_label(*condition)} leaves {denominator} of "
        f"{2 ** tosses} equally likely outcomes. {numerator} of those have "
        f"{_count_spec_label(*target)}, so the conditional probability is "
        f"{numerator}/{denominator}.",
        fraction,
    )


def _first_success_parity(question: str) -> Optional[str]:
    if not re.search(r"\buntil the first\b|\bfirst (?:head|success)\b", question, re.I):
        return None
    odd = bool(re.search(r"\bodd\b", question, re.I))
    even = bool(re.search(r"\beven\b", question, re.I))
    if not (odd or even):
        return None
    raw_p = _first_group(
        re.search(
            r"(?:probability|prob|p)\s*(?:of\s+success\s*)?(?:is|=|:)\s*"
            r"(\d+(?:\.\d+)?\s*%?)",
            question,
            re.I,
        )
    )
    if raw_p is not None:
        p = float(raw_p[:-1]) / 100 if raw_p.strip().endswith("%") else float(raw_p)
    elif re.search(r"\bcoin\b|\bhead\b", question, re.I):
        p = 0.5
    else:
        return None
    if not 0 < p < 1:
        return None
    failures = 1 - p
    denominator = 1 - failures ** 2
    value = p / denominator if odd else failures * p / denominator
    parity = "odd" if odd else "even"
    explanation = (
        f"The first success can occur on trial 1, 3, 5, ... with probabilities "
        f"p·q⁰, p·q², p·q⁴, ... where p={_fmt(p)} and q={_fmt(failures)}. "
        f"The geometric series sums to p/(1-q²) = {_fmt(value)} for an {parity} "
        f"number of trials."
    )
    return _prob(value, explanation)


def _de_mere(question: str) -> Optional[str]:
    if not re.search(r"\bdouble\s+six\b", question, re.I):
        return None
    if not re.search(r"\bat\s+least\s+one\s+six\b", question, re.I):
        return None
    single = _count_match(
        question, rf"\bat\s+least\s+one\s+six\s+in\s+({_NUMBER_TOKEN})\s+rolls?\b"
    )
    double = _count_match(
        question, rf"\b({_NUMBER_TOKEN})\s+rolls?\s+of\s+two\s+(?:fair\s+)?dice\b"
    )
    if single is None or double is None:
        return None
    first = 1 - (5 / 6) ** single
    second = 1 - (35 / 36) ** double
    if first > second:
        comparison = "The first is more likely."
    elif second > first:
        comparison = "The second is more likely."
    else:
        comparison = "They are equally likely."
    return (
        f"At least one six in {single} rolls: 1-(5/6)^{single} = {_fmt(first)}. "
        f"At least one double six in {double} rolls of two dice: "
        f"1-(35/36)^{double} = {_fmt(second)}. {comparison}"
    )


def _waiting_time(question: str) -> Optional[str]:
    if not re.search(r"\buntil\s+two\s+consecutive\s+heads\b", question, re.I):
        return None
    if not re.search(r"\bexpected\s+number\s+of\s+flips\b", question, re.I):
        return None
    return (
        "Let E₀ be the expected flips from no current head run, and E₁ from one "
        "current head. Then E₀ = 1 + ½E₀ + ½E₁ and E₁ = 1 + ½E₀ + ½·0. "
        "Solving gives E₀ = 6. The expected number of flips is 6."
    )


def _dice_sum_union(question: str) -> Optional[str]:
    if not re.search(r"\bdie\b|\bdice\b", question, re.I):
        return None
    match = re.search(
        rf"\bsum\s+(?:is|of)\s+({_NUMBER_TOKEN})\s+or\s+({_NUMBER_TOKEN})\b",
        question,
        re.I,
    )
    if not match:
        return None
    sides = _sides(question)
    if not sides or sides < 2:
        return None
    targets = [_count_from_text(match.group(1)), _count_from_text(match.group(2))]
    if any(target is None for target in targets):
        return None
    favourable = sum(
        sides - abs((sides + 1) - target)
        for target in targets
        if 2 <= target <= 2 * sides
    )
    total = sides ** 2
    return _prob(
        favourable / total,
        f"Two fair {sides}-sided dice give {total} ordered outcomes. The "
        f"targets are {targets[0]} and {targets[1]}, with {favourable} favourable "
        f"pairs, so P(sum is {targets[0]} or {targets[1]}) = {favourable}/{total}.",
    )


def _all_dice_equal(question: str) -> Optional[str]:
    if not re.search(r"\bdice\b", question, re.I):
        return None
    if not re.search(r"\ball\b|\bsame\b", question, re.I):
        return None
    count = _count_match(
        question, rf"\b({_NUMBER_TOKEN})\s+(?:fair\s+)?(?:six[- ]sided\s+)?dice\b"
    )
    sides = _sides(question)
    if count is None or count < 2 or not sides or sides < 2:
        return None
    total = sides ** count
    favourable = sides
    reduced = math.gcd(favourable, total)
    fraction = f"{favourable // reduced}/{total // reduced}"
    return _prob(
        favourable / total,
        f"There are {total} ordered outcomes from {count} fair {sides}-sided "
        f"dice. {favourable} have all dice showing the same face, so the "
        f"probability is {favourable}/{total}.",
        fraction,
    )


def _at_least_one_face(question: str) -> Optional[str]:
    if not re.search(r"\bdie\b|\bdice\b", question, re.I):
        return None
    if re.search(r"\bdouble\s+six\b", question, re.I):
        return None
    face = _count_match(
        question, rf"\bat\s+least\s+one\s+(?:die\s+)?(?:shows?\s+)?({_NUMBER_TOKEN})\b"
    )
    rolls = _count_match(question, rf"\brolled\s+({_NUMBER_TOKEN})\s+times?\b")
    if rolls is None:
        rolls = _count_match(question, rf"\b({_NUMBER_TOKEN})\s+rolls?\b")
    sides = _sides(question)
    if face is None or rolls is None or not sides or not 1 <= face <= sides:
        return None
    favourable = sides ** rolls - (sides - 1) ** rolls
    total = sides ** rolls
    reduced = math.gcd(favourable, total)
    fraction = f"{favourable // reduced}/{total // reduced}"
    value = favourable / total
    return _prob(
        value,
        f"P(at least one {face} in {rolls} rolls) = 1 - P(no {face}) = "
        f"1 - ({sides - 1}/{sides})^{rolls} = {favourable}/{total} = {_fmt(value)}.",
        fraction,
    )


def _die_order(question: str) -> Optional[str]:
    if not re.search(r"\bdie\b|\bdice\b", question, re.I):
        return None
    if not re.search(r"\bfirst\b|\bsecond\b", question, re.I):
        return None
    checks = (
        (
            re.search(
                r"\bsecond\b.*?\b(?:strictly\s+)?(?:greater|larger|more)\s+than\b.*?\bfirst\b",
                question,
                re.I,
            ),
            15,
            "second roll is strictly greater than the first",
        ),
        (
            re.search(
                r"\bfirst\b.*?\b(?:strictly\s+)?(?:greater|larger|more)\s+than\b.*?\bsecond\b",
                question,
                re.I,
            ),
            15,
            "first roll is strictly greater than the second",
        ),
        (
            re.search(
                r"\bsecond\b.*?\b(?:strictly\s+)?(?:less|smaller)\s+than\b.*?\bfirst\b",
                question,
                re.I,
            ),
            15,
            "second roll is strictly less than the first",
        ),
        (
            re.search(
                r"\bsecond\b.*?\bat\s+least\b.*?\bfirst\b", question, re.I
            ),
            21,
            "second roll is at least the first",
        ),
        (
            re.search(
                r"\bsecond\b.*?\bat\s+most\b.*?\bfirst\b", question, re.I
            ),
            21,
            "second roll is at most the first",
        ),
        (
            re.search(r"\bthe\s+two\s+rolls?\s+(?:are\s+)?(?:equal|the\s+same)\b", question, re.I),
            6,
            "two rolls are equal",
        ),
        (
            re.search(r"\b(?:equal|same)\b", question, re.I) and re.search(r"\bfirst\b.*?\bsecond\b", question, re.I),
            6,
            "two rolls are equal",
        ),
    )
    for match, favourable, label in checks:
        if match:
            return _prob(
                favourable / 36,
                f"Two ordered rolls give 6·6 = 36 equally likely pairs; "
                f"{favourable} have {label}.",
            )
    return None


def _binomial_probability(question: str) -> Optional[str]:
    if not re.search(r"\bbinomial\b", question, re.I):
        return None
    notation = re.search(
        r"binomial\s*\(\s*(\d+)\s*,\s*(\d+(?:\.\d+)?\s*%?)\s*\)", question, re.I
    )
    if notation:
        n = int(notation.group(1))
        raw_p = notation.group(2)
    else:
        n = _count_match(question, rf"\bn\s*(?:is|=|:)\s*({_NUMBER_TOKEN})\b")
        raw_p = _first_group(
            re.search(
                r"(?:success\s+probability|probability\s+of\s+success|\bp)\s*"
                r"(?:is|=|:)\s*(\d+(?:\.\d+)?\s*%?)",
                question,
                re.I,
            )
        )
    if n is None or n <= 0 or raw_p is None:
        return None
    p = _percent(raw_p)
    if not 0 <= p <= 1:
        return None

    ks: List[int] = []
    target_label = ""
    for match in re.finditer(r"p\s*\(\s*x\s*=\s*(\d+)\s*\)", question, re.I):
        ks.append(int(match.group(1)))
    if not ks:
        k_raw = _first_group(
            re.search(rf"\bexactly\s+({_NUMBER_TOKEN})\s+success(?:es)?\b", question, re.I)
        )
        parsed = _count_from_text(k_raw) if k_raw is not None else None
        if parsed is not None:
            ks = [parsed]
    if not ks:
        inequality = re.search(
            r"p\s*\(\s*x\s*(≥|>=|≤|<=|>|<)\s*(\d+)\s*\)", question, re.I
        )
        if inequality:
            symbol = inequality.group(1)
            threshold = int(inequality.group(2))
            if symbol in {"≥", ">="}:
                ks = list(range(threshold, n + 1))
                target_label = f"X ≥ {threshold}"
            elif symbol == ">":
                ks = list(range(threshold + 1, n + 1))
                target_label = f"X > {threshold}"
            elif symbol in {"≤", "<="}:
                ks = list(range(0, min(threshold, n) + 1))
                target_label = f"X ≤ {threshold}"
            else:
                ks = list(range(0, min(threshold - 1, n) + 1))
                target_label = f"X < {threshold}"
    if not ks or any(k < 0 or k > n for k in ks):
        return None

    terms = [
        math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in ks
    ]
    value = sum(terms)
    formula = " + ".join(
        f"C({n},{k})·{_fmt(p)}^{k}·{_fmt(1 - p)}^{n - k}" for k in ks
    )
    targets = target_label or " + ".join(f"P(X={k})" for k in ks)
    return _prob(
        value,
        f"For X ~ Binomial({n},{_fmt(p)}), {targets} = {formula} = {_fmt(value)}.",
    )


def _birthday_problem(question: str) -> Optional[str]:
    if not re.search(r"\bbirthday\b", question, re.I):
        return None
    people = _count_match(question, rf"\b({_NUMBER_TOKEN})\s+people\b")
    if people is None:
        people = _count_match(question, rf"\bn\s*(?:is|=|:)\s*({_NUMBER_TOKEN})\b")
    if people is None or people < 2:
        return None
    days_match = re.search(
        r"\b(\d+)\s+(?:equally\s+likely\s+)?(?:days|birthdays)\b", question, re.I
    )
    days = int(days_match.group(1)) if days_match else 365
    if days < 1:
        return None
    no_match = 1.0
    if people <= days:
        for offset in range(people):
            no_match *= (days - offset) / days
    else:
        no_match = 0.0
    value = 1 - no_match
    return _prob(
        value,
        f"With {days} equally likely birthdays, the chance that all {people} "
        f"people have distinct birthdays is Π(days-i)/days for i=0 to {people - 1} "
        f"= {_fmt(no_match)}. The complement is the chance of at least one shared "
        f"birthday.",
    )


def _uniform_distribution(question: str) -> Optional[str]:
    if not re.search(r"\buniform(?:ly)?\b", question, re.I):
        return None
    interval = re.search(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", question
    )
    if not interval:
        interval = re.search(
            r"\bbetween\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)\b",
            question,
            re.I,
        )
    if not interval:
        return None
    lower, upper = sorted((float(interval.group(1)), float(interval.group(2))))
    if upper <= lower:
        return None

    bounds = re.search(
        r"p\s*\(\s*(-?\d+(?:\.\d+)?)\s*(?:≤|<=)\s*x\s*(?:≤|<=)\s*"
        r"(-?\d+(?:\.\d+)?)\s*\)",
        question,
        re.I,
    )
    if not bounds:
        return None
    wanted_low, wanted_high = sorted((float(bounds.group(1)), float(bounds.group(2))))
    clipped_low, clipped_high = max(lower, wanted_low), min(upper, wanted_high)
    value = (clipped_high - clipped_low) / (upper - lower) if clipped_high > clipped_low else 0.0
    return _prob(
        value,
        f"For X ~ Uniform[{_fmt(lower)}, {_fmt(upper)}], probability is interval "
        f"length over total length: ({_fmt(clipped_high)} - {_fmt(clipped_low)})/"
        f"({_fmt(upper)} - {_fmt(lower)}) = {_fmt(value)}.",
    )


def _hypergeometric(question: str) -> Optional[str]:
    if not re.search(r"\bwithout replacement\b|\bhypergeometric\b", question, re.I):
        return None
    population = _count_match(
        question, rf"\b(?:batch|population|lot|box|group|container)\s+of\s+({_NUMBER_TOKEN})\b"
    )
    if population is None:
        population = _count_match(question, rf"\b({_NUMBER_TOKEN})\s+items?\b")
    successes = _count_match(
        question,
        rf"\b({_NUMBER_TOKEN})\s+(?:are\s+)?(?:defective|success(?:es)?|marked|red)\b",
    )
    draws = _count_match(
        question,
        rf"\b({_NUMBER_TOKEN})\s+items?\s+(?:are\s+)?(?:sampled|drawn|selected)\b",
    )
    if draws is None:
        draws = _count_match(question, rf"\b(?:sample|draw|select)\s+(?:of\s+)?({_NUMBER_TOKEN})\b")
    observed = _count_match(
        question,
        rf"\bexactly\s+({_NUMBER_TOKEN})\s+(?:are\s+)?(?:defective|success(?:es)?|marked|red)\b",
    )
    at_least = _count_match(
        question,
        rf"\bat\s+least\s+({_NUMBER_TOKEN})\s+(?:are\s+)?(?:defective|success(?:es)?|marked|red)\b",
    )
    if None in (population, successes, draws):
        return None
    if observed is None and at_least is None:
        return None
    if not (0 < draws <= population and 0 <= successes <= population):
        return None
    if observed is not None:
        value = _hypergeom_pmf(population, successes, draws, observed)
        detail = (
            f"C({successes},{observed})·C({population - successes},{draws - observed})/"
            f"C({population},{draws})"
        )
        label = f"exactly {observed}"
    else:
        ks = list(range(at_least, successes + 1))
        value = math.fsum(_hypergeom_pmf(population, successes, draws, k) for k in ks)
        detail = (
            f"Σ C({successes},k)·C({population - successes},{draws} - k)/"
            f"C({population},{draws})"
        )
        label = f"at least {at_least}"
    return _prob(
        value,
        f"Sampling without replacement is hypergeometric: P({label}) = {detail} = {_fmt(value)}.",
    )


def _variance_transformation(question: str) -> Optional[str]:
    var_match = re.search(
        r"var(?:iance)?\s*\(\s*([A-Za-z])\s*\)\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)",
        question,
        re.I,
    )
    transform = re.search(
        r"var(?:iance)?\s*\(\s*(\d+(?:\.\d+)?)\s*([A-Za-z])\s*"
        r"(?:[+-]\s*(\d+(?:\.\d+)?))?\s*\)",
        question,
        re.I,
    )
    if not (var_match and transform):
        return None
    if var_match.group(1).lower() != transform.group(2).lower():
        return None
    coefficient = float(transform.group(1))
    variance = float(var_match.group(2))
    shift = transform.group(3) or "0"
    value = coefficient ** 2 * variance
    return (
        f"For constants a and b, Var(aX + b) = a²Var(X). Here a={_fmt(coefficient)} "
        f"and b={_fmt(float(shift))}, so Var({_fmt(coefficient)}X + {_fmt(float(shift))}) "
        f"= {_fmt(coefficient)}²·{_fmt(variance)} = {_fmt(value)}."
    )


def _moments_calculation(question: str) -> Optional[str]:
    if not re.search(r"\bcorrelation\b|\bcoefficient\b", question, re.I):
        return None
    covariance = _labeled_probability(question, r"(?:cov|covariance)\s*\(\s*x\s*,\s*y\s*\)")
    var_x = _labeled_probability(question, r"var(?:iance)?\s*\(\s*x\s*\)")
    var_y = _labeled_probability(question, r"var(?:iance)?\s*\(\s*y\s*\)")
    if covariance is None or var_x is None or var_y is None:
        return None
    if var_x <= 0 or var_y <= 0:
        return None
    sd_x, sd_y = math.sqrt(var_x), math.sqrt(var_y)
    value = covariance / (sd_x * sd_y)
    return (
        f"r = Cov(X,Y)/(SD(X)·SD(Y)) = {_fmt(covariance)}/"
        f"({_fmt(sd_x)}·{_fmt(sd_y)}) = {_fmt(value)}."
    )


def _regression_line_parameters(question: str) -> Optional[str]:
    if not re.search(r"\bregression\b|\bline\b", question, re.I):
        return None
    points = re.findall(
        r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", question
    )
    if len(points) >= 2 and re.search(r"\bslope\b", question, re.I):
        x1, y1 = float(points[0][0]), float(points[0][1])
        x2, y2 = float(points[1][0]), float(points[1][1])
        if x2 == x1:
            return None
        slope = (y2 - y1) / (x2 - x1)
        return (
            f"slope = (y₂ - y₁)/(x₂ - x₁) = ({_fmt(y2)} - {_fmt(y1)})/"
            f"({_fmt(x2)} - {_fmt(x1)}) = {_fmt(slope)}."
        )
    point = re.search(
        r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", question
    )
    if not point:
        return None
    x, y = float(point.group(1)), float(point.group(2))
    slope = _labeled_probability(question, r"\bslope\b")
    if slope is not None and re.search(r"\bintercept\b", question, re.I):
        intercept = y - slope * x
        return (
            f"For ŷ = a + bx, a = y - b·x = {_fmt(y)} - {_fmt(slope)}·{_fmt(x)} "
            f"= {_fmt(intercept)}."
        )
    intercept = _labeled_probability(question, r"\bintercept\b")
    if intercept is not None and x != 0 and re.search(r"\bslope\b", question, re.I):
        derived_slope = (y - intercept) / x
        return (
            f"For ŷ = a + bx, b = (y - a)/x = ({_fmt(y)} - {_fmt(intercept)})/"
            f"{_fmt(x)} = {_fmt(derived_slope)}."
        )
    return None


def _bernoulli_trials(question: str) -> Optional[str]:
    n_match = re.search(r"\b(\d+)\s+(?:independent\s+)?(?:bernoulli\s+)?trials?\b", question, re.I)
    p_match = re.search(
        r"(?:success\s+probability|probability\s+of\s+success|\bp\s*)\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?\s*%?)",
        question,
        re.I,
    )
    if not (n_match and p_match):
        return None
    n = int(n_match.group(1))
    raw_p = p_match.group(1).strip()
    p = float(raw_p[:-1]) / 100 if raw_p.endswith("%") else float(raw_p)
    if not 0 <= p <= 1 or n <= 0:
        return None

    k_match = re.search(
        r"\b(exactly|at least|at most|no|zero)\s+(\d+|" + "|".join(_NUMBER_WORDS) + r")?\s*success(?:es)?\b",
        question,
        re.I,
    )
    mode = "exact"
    k = 1
    if k_match:
        raw_mode = k_match.group(1).lower()
        mode = "exact" if raw_mode in {"exactly", "no", "zero"} else raw_mode
        raw_k = k_match.group(2) or "1"
        parsed = _count_from_text(raw_k)
        if parsed is None:
            return None
        k = parsed
    else:
        x_match = re.search(r"p\s*\(\s*x\s*=\s*(\d+)\s*\)", question, re.I)
        if x_match:
            k = int(x_match.group(1))
        else:
            return None

    if mode in {"no", "zero"}:
        k, mode = 0, "exact"
    if k < 0:
        return None

    if mode == "exact":
        numerator = math.comb(n, k) if k <= n else 0
        value = numerator * (p ** k) * ((1 - p) ** (n - k)) if k <= n else 0.0
        label = f"exactly {k} success{'es' if k != 1 else ''}"
        formula = f"C({n},{k})·{_fmt(p)}^{k}·{_fmt(1 - p)}^{n - k}"
    elif mode == "at least":
        terms = range(k, n + 1)
        value = sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in terms)
        label = f"at least {k} success{'es' if k != 1 else ''}"
        formula = f"1 - P(X ≤ {k - 1}) for X ~ Binomial({n},{_fmt(p)})"
    else:
        terms = range(0, min(k, n) + 1)
        value = sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in terms)
        label = f"at most {k} success{'es' if k != 1 else ''}"
        formula = f"Σ from i=0 to {k} of C({n},i)·{_fmt(p)}^i·{_fmt(1 - p)}^({n}-i)"

    return _prob(
        value,
        f"With {n} independent trials and success probability {_fmt(p)}, P({label}) = {formula} = {_fmt(value)}.",
    )


def _poisson(question: str) -> Optional[str]:
    if not re.search(r"\bpoisson\b", question, re.I):
        return None
    lam_match = re.search(
        r"(?:lambda|λ|mean)\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)", question, re.I
    )
    if not lam_match:
        return None
    lam = float(lam_match.group(1))
    pmf = lambda k: math.exp(-lam) * (lam ** k) / math.factorial(k)

    k_match = re.search(r"p\s*\(\s*x\s*=\s*(\d+)\s*\)", question, re.I)
    if k_match:
        k = int(k_match.group(1))
        return _prob(
            pmf(k),
            f"For X ~ Poisson(λ={_fmt(lam)}), P(X={k}) = e^(-λ)·λ^{k}/{k}! = {_fmt(pmf(k))}.",
        )

    inequality = re.search(
        r"p\s*\(\s*x\s*(≤|<=|<|≥|>=|>)\s*(\d+)\s*\)", question, re.I
    )
    if not inequality:
        return None
    symbol, threshold = inequality.group(1), int(inequality.group(2))
    if symbol in {"≤", "<="}:
        ks = list(range(threshold + 1))
        label = f"X ≤ {threshold}"
    elif symbol == "<":
        ks = list(range(threshold))
        label = f"X < {threshold}"
    elif symbol in {"≥", ">="}:
        ks = list(range(threshold, threshold + 40))
        label = f"X ≥ {threshold}"
    else:
        ks = list(range(threshold + 1, threshold + 41))
        label = f"X > {threshold}"
    value = math.fsum(pmf(k) for k in ks)
    return _prob(
        value,
        f"For X ~ Poisson(λ={_fmt(lam)}), P({label}) = Σ e^(-λ)·λ^k/k! "
        f"over the requested k values = {_fmt(value)}.",
    )


def _geometric(question: str) -> Optional[str]:
    if not re.search(r"\bgeometric\b", question, re.I):
        return None
    p_match = re.search(r"\bp\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)", question, re.I)
    k_match = re.search(
        r"(?:first success occurs on|trial)\s+(\d+)|p\s*\(\s*x\s*=\s*(\d+)\s*\)",
        question,
        re.I,
    )
    if not (p_match and k_match):
        return None
    p = float(p_match.group(1))
    k = int(k_match.group(1) or k_match.group(2))
    if not 0 < p <= 1 or k < 1:
        return None
    value = ((1 - p) ** (k - 1)) * p
    return _prob(
        value,
        f"For a geometric distribution with p={_fmt(p)}, P(first success on trial {k}) = (1-p)^({k - 1})·p = {_fmt(value)}.",
    )


def _cdf_interval(question: str) -> Optional[str]:
    cdf = re.search(
        r"f\s*\(\s*x\s*\)\s*=\s*1\s*-\s*e\s*\^\s*\(\s*-?\s*(\d+(?:\.\d+)?)\s*x\s*\)",
        question,
        re.I,
    )
    bounds = re.search(
        r"p\s*\(\s*(-?\d+(?:\.\d+)?)\s*<\s*x\s*(?:≤|<=|<)\s*(-?\d+(?:\.\d+)?)\s*\)",
        question,
        re.I,
    )
    if not (cdf and bounds):
        return None
    rate = float(cdf.group(1))
    low, high = float(bounds.group(1)), float(bounds.group(2))
    if high <= low:
        return None
    value = math.exp(-rate * low) - math.exp(-rate * high)
    return _prob(
        value,
        f"F(x) = 1 - e^(-{_fmt(rate)}x), so P({_fmt(low)} < X ≤ {_fmt(high)}) = "
        f"F({_fmt(high)}) - F({_fmt(low)}) = e^(-{_fmt(rate * low)}) - "
        f"e^(-{_fmt(rate * high)}) = {_fmt(value)}.",
    )


def _exponential(question: str) -> Optional[str]:
    if not re.search(r"\bexponential\b", question, re.I) or not re.search(
        r"\brate\b|\bwaiting\b|\bdistribution\b", question, re.I
    ):
        return None
    rate_match = re.search(r"\brate\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)", question, re.I)
    bound = re.search(
        r"p\s*\(\s*x\s*(>|<|≥|≤|>=|<=)\s*(\d+(?:\.\d+)?)\s*\)|after\s+(\d+(?:\.\d+)?)",
        question,
        re.I,
    )
    if not (rate_match and bound):
        return None
    rate = float(rate_match.group(1))
    symbol = bound.group(1)
    if symbol:
        x = float(bound.group(2))
        upper = symbol in {"<", "≤", "<="}
        value = 1 - math.exp(-rate * x) if upper else math.exp(-rate * x)
        relation = "<" if upper else ">"
        explanation = (
            f"For an exponential waiting time with rate λ={_fmt(rate)}, "
            f"P(X {relation} {_fmt(x)}) = {'1 - ' if upper else ''}e^(-λx) = "
            f"{_fmt(value)}."
        )
    else:
        x = float(bound.group(3))
        value = math.exp(-rate * x)
        explanation = (
            f"For an exponential waiting time with rate λ={_fmt(rate)}, "
            f"P(X > {_fmt(x)}) = e^(-λx) = {_fmt(value)}."
        )
    return _prob(value, explanation)


def _normal_parameters(question: str) -> Optional[Tuple[float, float]]:
    mean = _labeled_probability(question, r"\b(?:mean|mu|μ)\b")
    sd = _labeled_probability(question, r"\b(?:standard\s+deviation|sigma|σ)\b")
    if mean is None or sd is None or sd <= 0:
        return None
    return mean, sd


def _normal(question: str) -> Optional[str]:
    if not re.search(r"\bnormal(?:ly)?\b|\bz-score\b|\bz\s+statistic\b", question, re.I):
        return None
    params = _normal_parameters(question)
    if not params:
        return None
    mean, sd = params
    z_match = re.search(r"z-score\s+for\s+(\d+(?:\.\d+)?)", question, re.I)
    interval_match = re.search(
        r"p\s*\(\s*(-?\d+(?:\.\d+)?)\s*<\s*x\s*<\s*(-?\d+(?:\.\d+)?)\s*\)",
        question,
        re.I,
    )
    prob_match = re.search(
        r"p\s*\(\s*x\s*(<|>)\s*(\d+(?:\.\d+)?)\s*\)", question, re.I
    )
    if interval_match:
        low, high = float(interval_match.group(1)), float(interval_match.group(2))
        z_low, z_high = (low - mean) / sd, (high - mean) / sd
        cdf = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
        value = cdf(z_high) - cdf(z_low)
        return _prob(
            value,
            f"Standardise the bounds: z = ({_fmt(low)} - {_fmt(mean)})/{_fmt(sd)} "
            f"= {_fmt(z_low)} and z = ({_fmt(high)} - {_fmt(mean)})/{_fmt(sd)} "
            f"= {_fmt(z_high)}. So P({_fmt(low)} < X < {_fmt(high)}) = "
            f"Φ({_fmt(z_high)}) - Φ({_fmt(z_low)}) = {_fmt(value)}.",
        )
    if z_match:
        x = float(z_match.group(1))
        z = (x - mean) / sd
        return f"z = (x - μ)/σ = ({_fmt(x)} - {_fmt(mean)})/{_fmt(sd)} = {_fmt(z)}."
    if prob_match:
        direction = prob_match.group(1)
        x = float(prob_match.group(2))
        z = (x - mean) / sd
        cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        value = cdf if direction == "<" else 1 - cdf
        relation = "below" if direction == "<" else "above"
        return _prob(
            value,
            f"Standardise x={_fmt(x)}: z = ({_fmt(x)} - {_fmt(mean)})/{_fmt(sd)} = {_fmt(z)}. "
            f"P(X {direction} {_fmt(x)}) = {_fmt(value)} ({relation} the mean).",
        )
    return None


def _discrete_table(question: str) -> Optional[str]:
    match = re.search(
        r"values\s+([0-9]+(?:\.[0-9]+)?(?:\s*,\s*[0-9]+(?:\.[0-9]+)?)+)\s+"
        r"with\s+probabilities\s+([0-9]+(?:\.[0-9]+)?(?:\s*,\s*[0-9]+(?:\.[0-9]+)?)+)",
        question,
        re.I,
    )
    if not match:
        return None
    values = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", match.group(1))]
    probs = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", match.group(2))]
    if len(values) != len(probs) or not probs or abs(sum(probs) - 1) > 1e-6:
        return None
    expected = sum(value * prob for value, prob in zip(values, probs))
    expected_square = sum((value ** 2) * prob for value, prob in zip(values, probs))
    variance = expected_square - expected ** 2
    if re.search(r"\bvariance\b|\bvar\s*\(", question, re.I):
        return (
            f"E[X] = {_fmt(expected)}, E[X²] = {_fmt(expected_square)}, so "
            f"Var(X) = E[X²] - (E[X])² = {_fmt(expected_square)} - {_fmt(expected)}² = {_fmt(variance)}."
        )
    return (
        f"E[X] = Σx·P(X=x) = {_fmt(expected)}. "
        f"For reference, Var(X) = {_fmt(variance)}."
    )


def _fair_die_moments(question: str) -> Optional[str]:
    if not re.search(r"\bfair\s+(?:six-sided\s+)?die\b|\bfair\s+dice\b", question, re.I):
        return None
    if re.search(r"\bvariance\b|\bvar\s*\(", question, re.I):
        return (
            "For a fair six-sided die, E[X] = (1+2+3+4+5+6)/6 = 3.5 and "
            "Var(X) = 35/12 ≈ 2.9167."
        )
    if re.search(r"\bexpected\s+value\b|\bexpectation\b|\bmean\b", question, re.I):
        return "For a fair six-sided die, E[X] = (1+2+3+4+5+6)/6 = 3.5."
    return None


def _two_dice(question: str) -> Optional[str]:
    if not re.search(r"\btwo\s+(?:fair\s+)?dice\b|\b2\s+(?:fair\s+)?dice\b", question, re.I):
        return None
    if re.search(r"\bboth\s+(?:are\s+)?even\b", question, re.I):
        return _prob(9 / 36, "Each die has 3 even faces, so 3·3 = 9 of the 36 ordered outcomes are even-even.", "1/4")
    sum_match = re.search(r"\bsum\s+is\s+(\d+)\b|\bsum\s+of\s+(\d+)\b", question, re.I)
    if sum_match:
        target = int(sum_match.group(1) or sum_match.group(2))
        if 2 <= target <= 12:
            favourable = 6 - abs(7 - target)
            return _prob(
                favourable / 36,
                f"There are {favourable} ordered pairs of two dice that sum to {target}, out of 36 equally likely outcomes.",
            )
    return None


def _multi_roll_sum(question: str) -> Optional[str]:
    """Exact or inequality probabilities for the sum of repeated fair dice."""
    if not re.search(r"\bsum\b", question, re.I):
        return None
    if not re.search(r"\bdie\b|\bdice\b", question, re.I):
        return None
    if re.search(r"\bbiased\b|\bunfair\b|\bloaded\b", question, re.I):
        return None

    roll_match = re.search(
        rf"\b(?:rolled|rolls?|throws?|tossed)\s+({_NUMBER_TOKEN})\s+times?\b",
        question,
        re.I,
    )
    if not roll_match:
        return None
    rolls = _count_from_text(roll_match.group(1))
    sides = _sides(question)
    if rolls is None or rolls <= 0 or sides is None or sides < 2:
        return None

    exact = re.search(rf"\bsum\s+(?:is|of|equal\s+to|=)\s*({_NUMBER_TOKEN})\b", question, re.I)
    at_least = re.search(
        rf"\bsum\s+(?:is\s+)?at\s+least\s*({_NUMBER_TOKEN})\b|\bsum\s*≥\s*({_NUMBER_TOKEN})\b",
        question,
        re.I,
    )
    at_most = re.search(
        rf"\bsum\s+(?:is\s+)?at\s+most\s*({_NUMBER_TOKEN})\b|\bsum\s*≤\s*({_NUMBER_TOKEN})\b",
        question,
        re.I,
    )
    greater = re.search(
        rf"\bsum\s+(?:is\s+)?(?:greater|more)\s+than\s*({_NUMBER_TOKEN})\b|\bsum\s*>\s*({_NUMBER_TOKEN})\b",
        question,
        re.I,
    )
    less = re.search(
        rf"\bsum\s+(?:is\s+)?(?:less|below)\s+than\s*({_NUMBER_TOKEN})\b|\bsum\s*<\s*({_NUMBER_TOKEN})\b",
        question,
        re.I,
    )

    relation: Optional[str] = None
    threshold: Optional[int] = None
    label = ""
    if exact:
        parsed = _count_from_text(exact.group(1))
        if parsed is not None:
            relation, threshold, label = "exact", parsed, f"exactly {parsed}"
    elif at_least:
        parsed = _count_from_text(_first_group(at_least))
        if parsed is not None:
            relation, threshold, label = "at_least", parsed, f"at least {parsed}"
    elif greater:
        parsed = _count_from_text(_first_group(greater))
        if parsed is not None:
            relation, threshold, label = "at_least", parsed + 1, f"greater than {parsed}"
    elif at_most:
        parsed = _count_from_text(_first_group(at_most))
        if parsed is not None:
            relation, threshold, label = "at_most", parsed, f"at most {parsed}"
    elif less:
        parsed = _count_from_text(_first_group(less))
        if parsed is not None:
            relation, threshold, label = "at_most", parsed - 1, f"less than {parsed}"
    if relation is None or threshold is None:
        return None

    lowest, highest = rolls, rolls * sides
    if relation == "exact" and (threshold < lowest or threshold > highest):
        return (
            f"The smallest possible sum from {rolls} roll{'s' if rolls != 1 else ''} "
            f"is {lowest} and the largest is {highest}, so a sum of {threshold} "
            f"is impossible: probability 0."
        )

    # Count ordered outcomes over the full generating polynomial
    # (x + x^2 + ... + x^sides)^rolls.
    ways = [0] * (highest + 1)
    ways[0] = 1
    for _ in range(rolls):
        updated = [0] * (highest + 1)
        for current, count in enumerate(ways):
            if not count:
                continue
            for face in range(1, sides + 1):
                if current + face <= highest:
                    updated[current + face] += count
        ways = updated

    if relation == "exact":
        favourable = ways[threshold]
    else:
        lower = max(lowest, threshold) if relation == "at_least" else lowest
        upper = highest if relation == "at_least" else min(highest, threshold)
        favourable = sum(ways[lower:upper + 1]) if lower <= upper else 0

    total = sides ** rolls
    reduced = math.gcd(favourable, total)
    fraction = f"{favourable // reduced}/{total // reduced}" if favourable else None
    return _prob(
        favourable / total,
        f"A fair {sides}-sided die has {sides} equally likely faces per roll, so "
        f"{rolls} roll{'s' if rolls != 1 else ''} give {sides}^{rolls} = {total} ordered "
        f"outcomes. {favourable} of them have a sum {label}, so "
        f"P(sum {label}) = {favourable}/{total}.",
        fraction,
    )


def _cards(question: str) -> Optional[str]:
    if not re.search(r"\bcard\b|\bdeck\b", question, re.I):
        return None
    rank_counts = {
        "ace": 4, "king": 4, "queen": 4, "jack": 4,
        "spade": 13, "heart": 13, "club": 13, "diamond": 13,
        "red": 26, "black": 26,
    }
    found = [
        (name, count)
        for name, count in rank_counts.items()
        if re.search(rf"\b{name}\b", question, re.I)
    ]
    if not found:
        return None

    draw_match = re.search(rf"\b({_NUMBER_TOKEN})\s+cards?\b", question, re.I)
    draws = _count_from_text(draw_match.group(1)) if draw_match else 1
    if draws < 1:
        return None
    replacement = bool(re.search(r"\bwith replacement\b", question, re.I))

    if re.search(r"\bthen\b", question, re.I) and len(found) >= 2:
        first, second = found[0][1], found[1][1]
        denominator = 52 if replacement else 51
        value = (first / 52) * (second / denominator)
        return _prob(
            value,
            f"P(first {found[0][0]}, then {found[1][0]}) = {first}/52 · "
            f"{second}/{denominator} = {_fmt(value)}.",
        )

    name, count = found[0]
    if re.search(r"\bat\s+least\s+one\b", question, re.I):
        if replacement:
            value = 1 - ((52 - count) / 52) ** draws
            note = f"1 - ({52 - count}/52)^{draws}"
        else:
            no_target = 1.0 if draws > 52 - count else math.comb(52 - count, draws) / math.comb(52, draws)
            value = 1 - no_target
            note = f"1 - C({52 - count},{draws})/C(52,{draws})"
        return _prob(
            value,
            f"P(at least one {name} in {draws} card{'s' if draws != 1 else ''}) = {note} = {_fmt(value)}.",
        )

    exact_match = re.search(rf"\bexactly\s+({_NUMBER_TOKEN})\b", question, re.I)
    both_match = re.search(
        rf"\b(?:both|all)\s+(?:are\s+)?{name}\b|\bboth\s+{name}\b", question, re.I
    )
    if exact_match or both_match:
        k = _count_from_text(exact_match.group(1)) if exact_match else draws
        if k is None or k > draws:
            return None
        if replacement:
            value = (
                math.comb(draws, k)
                * (count / 52) ** k
                * ((52 - count) / 52) ** (draws - k)
            )
            formula = f"C({draws},{k})·({count}/52)^{k}·({52 - count}/52)^{draws - k}"
        else:
            value = _hypergeom_pmf(52, count, draws, k)
            formula = f"C({count},{k})·C({52 - count},{draws - k})/C(52,{draws})"
        return _prob(
            value,
            f"For {draws} card{'s' if draws != 1 else ''} drawn "
            f"{'with' if replacement else 'without'} replacement, "
            f"P(exactly {k} {name}) = {formula} = {_fmt(value)}.",
        )

    if draws == 1:
        total = sum(count for _name, count in found)
        names = " or ".join(name for name, _count in found)
        return _prob(total / 52, f"A standard deck has 52 cards; {names} gives {total} favourable cards out of 52.")
    return None


def _urn(question: str) -> Optional[str]:
    urn = re.search(r"\b(\d+)\s+red\s*(?:balls?|marbles?)?\s+and\s+(\d+)\s+blue\s*(?:balls?|marbles?)?", question, re.I)
    if urn:
        red, blue = int(urn.group(1)), int(urn.group(2))
        total = red + blue
        if total < 2:
            return None
        without = bool(re.search(r"\bwithout replacement\b", question, re.I))
        draws = _count_match(
            question, rf"\b({_NUMBER_TOKEN})\s+balls?\s+(?:are\s+)?drawn\b"
        )
        exact_red = _count_match(
            question, rf"\bexactly\s+({_NUMBER_TOKEN})\s+(?:are\s+)?red\b"
        )
        if draws is not None and exact_red is not None and without:
            if 0 < draws <= total and 0 <= exact_red <= min(red, draws):
                value = _hypergeom_pmf(total, red, draws, exact_red)
                return _prob(
                    value,
                    f"Without replacement, the number of red balls is hypergeometric: "
                    f"C({red},{exact_red})·C({blue},{draws - exact_red})/"
                    f"C({total},{draws}) = {_fmt(value)}.",
                )
        if re.search(r"\bsame\s+colou?r\b", question, re.I):
            if without:
                favourable = math.comb(red, 2) + math.comb(blue, 2)
                denominator = math.comb(total, 2)
                return _prob(
                    favourable / denominator,
                    f"Without replacement, P(same colour) = "
                    f"[C({red},2)+C({blue},2)]/C({total},2) = {_fmt(favourable / denominator)}.",
                )
            value = (red / total) ** 2 + (blue / total) ** 2
            return _prob(
                value,
                f"With replacement, P(same colour) = ({red}/{total})² + "
                f"({blue}/{total})² = {_fmt(value)}.",
            )
        if re.search(r"\bboth\s+(?:are\s+)?red\b", question, re.I):
            if without:
                value = (red / total) * ((red - 1) / (total - 1))
                return _prob(value, f"Without replacement, P(both red) = {red}/{total}·{red - 1}/{total - 1} = {_fmt(value)}.")
            value = (red / total) ** 2
            return _prob(value, f"With replacement, P(both red) = ({red}/{total})² = {_fmt(value)}.")
        if re.search(r"\bexactly\s+one\s+(?:is\s+)?red\b", question, re.I):
            if without:
                value = red * blue / math.comb(total, 2)
                return _prob(value, f"Without replacement, P(exactly one red) = {red}·{blue}/C({total},2) = {_fmt(value)}.")
            value = 2 * (red / total) * (blue / total)
            return _prob(value, f"With replacement, P(exactly one red) = 2·{red}/{total}·{blue}/{total} = {_fmt(value)}.")
        if re.search(r"\bat\s+least\s+one\s+(?:is\s+)?red\b", question, re.I):
            value = 1 - (math.comb(blue, 2) / math.comb(total, 2) if without and blue >= 2 else (blue / total) ** 2)
            return _prob(value, f"P(at least one red) = 1 - P(no red) = {_fmt(value)}.")

    colors = re.search(r"\b(\d+)\s+(\w+)\s*(?:balls?|marbles?)?\s+and\s+(\d+)\s+(\w+)\b", question, re.I)
    if colors and re.search(r"\bthen\b", question, re.I):
        first_n, first_color, second_n, second_color = (
            int(colors.group(1)), colors.group(2).lower(), int(colors.group(3)), colors.group(4).lower()
        )
        if re.search(rf"\b{re.escape(first_color)}\s+then\s+{re.escape(second_color)}\b", question, re.I):
            total = first_n + second_n
            if total < 2:
                return None
            denominator = total if re.search(r"\bwith replacement\b", question, re.I) else total - 1
            second_count = second_n if re.search(r"\bwith replacement\b", question, re.I) else second_n
            value = (first_n / total) * (second_count / denominator)
            return _prob(
                value,
                f"P({first_color} then {second_color}) = {first_n}/{total}·{second_count}/{denominator} = {_fmt(value)}.",
            )
    return None


def _counting_questions(question: str) -> Optional[str]:
    perm = re.search(
        r"\bpermutations?\b.*?\bof\s+(\d+)\s+distinct\b", question, re.I
    )
    if perm:
        n = int(perm.group(1))
        value = math.factorial(n)
        return f"There are {n}! = {value} permutations of {n} distinct objects."

    choose = re.search(
        r"\bchoose\s+(\d+)\s+(?:objects?|items?|students?|people)?\s*from\s+(\d+)\b",
        question,
        re.I,
    )
    if choose:
        k, n = int(choose.group(1)), int(choose.group(2))
        if 0 <= k <= n:
            return (
                f"C({n},{k}) = {n}! / ({k}!·({n - k})!) = {math.comb(n, k)}. "
                f"Order does not matter when you choose."
            )

    digits = re.search(
        r"\bhow\s+many\s+(\d+)\s+digit\s+numbers?\s+can\s+be\s+formed\s+(?:by|from)\s+([0-9,\s]+?)(?=\s+with\b|\s+without\b|\s+what\b|\?|$)",
        question,
        re.I,
    )
    if digits and re.search(r"\bwithout repetition\b|\bwithout\s+repeating\b", question, re.I):
        length = int(digits.group(1))
        available = [int(value) for value in re.findall(r"\d", digits.group(2))]
        if not available or length > len(available):
            return None
        first_choices = [value for value in available if value != 0] or available
        if 0 in available:
            total = len(first_choices) * math.perm(len(available) - 1, length - 1)
            note = " The first digit cannot start with 0."
        else:
            total = math.perm(len(available), length)
            note = ""
        repeat_total = len(available) ** length
        factors = " × ".join(
            [str(len(first_choices))] + [str(len(available) - index) for index in range(1, length)]
        )
        return (
            f"{factors} = {total}.{note} If repetition were allowed, it would be "
            f"{len(available)}^{length} = {repeat_total}."
        )
    return None


def _confidence_interval(question: str) -> Optional[str]:
    if not re.search(r"\bconfidence interval\b|\bmargin of error\b", question, re.I):
        return None
    n_match = re.search(r"\bsample\s+of\s+(\d+)|\bn\s*=\s*(\d+)", question, re.I)
    mean_match = re.search(r"\bmean\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)", question, re.I)
    sd_match = re.search(r"standard\s+deviation\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)", question, re.I)
    level_match = re.search(r"(\d+)\s*%\s*confidence", question, re.I)
    if not (n_match and mean_match and sd_match and level_match):
        return None
    n = int(n_match.group(1) or n_match.group(2))
    mean = float(mean_match.group(1))
    sd = float(sd_match.group(1))
    level = int(level_match.group(1))
    z_values = {90: 1.645, 95: 1.96, 99: 2.576}
    z = z_values.get(level)
    if z is None or n <= 0:
        return None
    margin = z * sd / math.sqrt(n)
    lower, upper = mean - margin, mean + margin
    return (
        f"For a {level}% confidence interval with known σ, use x̄ ± z·σ/√n. "
        f"z={z}, σ={_fmt(sd)}, n={n}, so the margin of error is {_fmt(margin)}. "
        f"The interval is [{_fmt(lower)}, {_fmt(upper)}]."
    )


def _z_test(question: str) -> Optional[str]:
    if not re.search(r"\bh_?0\b|\bnull hypothesis\b|\bz\s+statistic\b|\bz-test\b", question, re.I):
        return None
    mu_match = re.search(r"(?:mu|μ|h_?0\s*:?\s*mu)\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)", question, re.I)
    mean_match = re.search(r"sample\s+mean\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)", question, re.I)
    sd_match = re.search(r"standard\s+deviation\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)", question, re.I)
    n_match = re.search(r"\bn\s*=\s*(\d+)", question, re.I)
    if not (mu_match and mean_match and sd_match and n_match):
        return None
    mu, xbar, sd, n = float(mu_match.group(1)), float(mean_match.group(1)), float(sd_match.group(1)), int(n_match.group(1))
    if n <= 0 or sd <= 0:
        return None
    z = (xbar - mu) / (sd / math.sqrt(n))
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    decision = "reject" if p_value < 0.05 else "do not reject"
    return (
        f"z = (x̄ - μ₀)/(σ/√n) = ({_fmt(xbar)} - {_fmt(mu)})/({_fmt(sd)}/√{n}) = {_fmt(z)}. "
        f"The two-sided p-value is {_fmt(p_value)}, so at 5% you {decision} the null hypothesis."
    )


def _array(label: str, question: str) -> Optional[List[float]]:
    match = re.search(rf"\b{label}\s*=\s*\[([^\]]+)\]", question, re.I)
    if not match:
        return None
    values = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", match.group(1))]
    return values or None


def _regression_data(question: str) -> Optional[str]:
    xs, ys = _array("x", question), _array("y", question)
    if not xs or not ys or len(xs) != len(ys) or len(xs) < 2:
        return None
    n = len(xs)
    xbar, ybar = sum(xs) / n, sum(ys) / n
    sxx = sum((value - xbar) ** 2 for value in xs)
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    if sxx <= 0:
        return None
    if re.search(r"\bcorrelation\b|\bpearson\b|\br\b", question, re.I):
        syy = sum((value - ybar) ** 2 for value in ys)
        r = sxy / math.sqrt(sxx * syy) if syy > 0 else 0.0
        return f"The Pearson correlation is r = Σ(x-x̄)(y-ȳ)/√(Σ(x-x̄)²Σ(y-ȳ)²) = {_fmt(r)}."
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    return (
        f"The least-squares line is ŷ = {_fmt(intercept)} + {_fmt(slope)}x. "
        f"slope = Σ(x-x̄)(y-ȳ)/Σ(x-x̄)² = {_fmt(slope)}; intercept = ȳ - slope·x̄ = {_fmt(intercept)}."
    )


_SOLVERS = (
    _event_algebra,
    _bayes_medical,
    _de_mere,
    _conditional_probability,
    _waiting_time,
    _first_success_parity,
    _dice_sum_union,
    _all_dice_equal,
    _at_least_one_face,
    _die_order,
    _poisson,
    _geometric,
    _binomial_probability,
    _birthday_problem,
    _uniform_distribution,
    _hypergeometric,
    _cdf_interval,
    _exponential,
    _normal,
    _two_dice,
    _multi_roll_sum,
    _cards,
    _urn,
    _discrete_table,
    _fair_die_moments,
    _bernoulli_trials,
    _variance_transformation,
    _moments_calculation,
    _confidence_interval,
    _z_test,
    _regression_line_parameters,
    _regression_data,
    _counting_questions,
)


def _run_solvers(question: str) -> Optional[str]:
    for solver in _SOLVERS:
        answer = solver(question)
        if answer:
            return answer
    return None


def _numeric_answer(question: str) -> Optional[str]:
    q = str(question or "").strip()
    if not q:
        return None
    # Let the established coin/toss helpers keep their precise wording.
    if re.search(r"\bcoin\b|\bheads?\b|\btails?\b|\btoss(?:es|ed|ing)?\b|\bflip(?:s|ped|ping)?\b", q, re.I):
        return None
    return _run_solvers(q)


def hard_answer(question: str) -> Optional[str]:
    """Run the numeric solvers even when coin/toss wording is present.

    Conditional and stopping-time questions were previously captured by the
    short coin helper, which answered the conditioning event instead of the
    requested probability. Calling this before the legacy helper closes that
    loophole while leaving ordinary coin questions untouched.
    """
    text = str(question or "").strip()
    if not text:
        return None
    low = _normalize(text)
    if not _TOPIC_GATE.search(low):
        return None
    return _run_solvers(low)


def answer(question: str) -> Optional[str]:
    """Return a broad probability/statistics answer when appropriate."""
    text = str(question or "").strip()
    if not text:
        return None
    # Normalise common probability/statistics typos before routing. The gateway
    # typo corrector in graph_rag handles conversational typos, while this one
    # keeps domain misspellings inside the maths engine.
    low = _normalize(text)
    if not _TOPIC_GATE.search(low):
        return None
    numeric = _numeric_answer(low)
    if numeric:
        return numeric
    return _concept_answer(low)
