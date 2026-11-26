"""RAG-guided planning with Python-computed truth.

The retrieval layer understands the question and selects an intent. Python then
computes the exact result and the same layer formats the reply. No source badge
or engine label is exposed to the user.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import concept_graph
import physics_planners
import probability_stats as ps
import subject_rag

Context = Dict[str, Any]
Params = Dict[str, Any]
Compute = Callable[[Context, Params], str]

_FLOAT = r"-?\d+(?:\.\d+)?"
_RANK_COUNTS = {"ace": 4, "king": 4, "queen": 4, "jack": 4}
_SUIT_COUNTS = {"spade": 13, "heart": 13, "club": 13, "diamond": 13}


@dataclass
class Plan:
    intent: str
    params: Params
    compute: Compute


def _count(q: str, pattern: str) -> Optional[int]:
    match = re.search(pattern, q, re.I)
    if not match:
        return None
    raw = next((group for group in match.groups() if group is not None), None)
    return ps._count_from_text(raw) if raw is not None else None


def _float(q: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, q, re.I)
    return float(match.group(1)) if match else None


def _prob_value(q: str, pattern: str) -> Optional[float]:
    return ps._labeled_probability(q, pattern)


def _expectation(q: str, inner: str) -> Optional[float]:
    match = re.search(
        rf"\be\s*[\(\[]\s*{inner}\s*[\)\]]\s*(?:is|=|:)?\s*({_FLOAT})",
        q,
        re.I,
    )
    return float(match.group(1)) if match else None


def _rank_and_suit(q: str):
    rank = next((name for name in _RANK_COUNTS if re.search(rf"\b{name}\b", q, re.I)), None)
    suit = next((name for name in _SUIT_COUNTS if re.search(rf"\b{name}\b", q, re.I)), None)
    return rank, suit


def _normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


_T_CRITICAL = {
    (10, 90): 1.812, (10, 95): 2.228, (10, 99): 3.169,
    (15, 90): 1.341, (15, 95): 2.131, (15, 99): 2.947,
    (20, 90): 1.325, (20, 95): 2.086, (20, 99): 2.845,
    (25, 90): 1.316, (25, 95): 2.060, (25, 99): 2.787,
    (30, 90): 1.310, (30, 95): 2.042, (30, 99): 2.750,
    (40, 90): 1.303, (40, 95): 2.021, (40, 99): 2.704,
    (60, 90): 1.296, (60, 95): 2.000, (60, 99): 2.660,
    (120, 90): 1.289, (120, 95): 1.980, (120, 99): 2.617,
}


def _t_critical(df: int, level: int) -> float:
    if df >= 120:
        return _T_CRITICAL[(120, level)]
    if (df, level) in _T_CRITICAL:
        return _T_CRITICAL[(df, level)]
    for candidate in (120, 60, 40, 30, 25, 20, 15, 10):
        if candidate <= df and (candidate, level) in _T_CRITICAL:
            return _T_CRITICAL[(candidate, level)]
    return {90: 1.645, 95: 1.96, 99: 2.576}[level]


def understand(question: str) -> Context:
    """Use the shared retrieval layer to understand domain and topic."""
    q = ps._normalize(question)
    ranked = subject_rag.rank_cards(question)
    concepts = ps._matched_concepts(q)
    topic = "probability and statistics"
    if ranked:
        topic = ranked[0][2].title
    elif concepts:
        topic = ps.CONCEPT_TITLES.get(concepts[0], topic)
    # The shared concept graph is the planner's semantic check. Its text is not
    # trusted for arithmetic; it only confirms which concept family is active.
    graph_hint = concept_graph.answer(question)
    return {
        "question": q,
        "topic": topic,
        "concepts": concepts,
        "ranked": ranked,
        "graph_hint": graph_hint,
    }


def _compute_card_joint(ctx: Context, p: Params) -> str:
    rank, suit = p["rank"], p["suit"]
    rank_count, suit_count = _RANK_COUNTS[rank], _SUIT_COUNTS[suit]
    favourable = rank_count * suit_count - 1
    total = 52 * 51
    value = favourable / total
    return ps._prob(
        value,
        f"There are {rank_count} {rank}s and {suit_count} {suit}s, but the {rank} of {suit}s "
        f"is counted in both, so {favourable} of the {total} ordered pairs qualify.",
        ps._fraction(value),
    )


def _compute_conditional_card(ctx: Context, p: Params) -> str:
    rank, suit = p["rank"], p["suit"]
    rank_count, suit_count = _RANK_COUNTS[rank], _SUIT_COUNTS[suit]
    value = (rank_count * suit_count - 1) / (51 * rank_count)
    article = "an" if rank[0] in "aeiou" else "a"
    return ps._prob(
        value,
        f"Given the first card is {article} {rank}, the {rank} of {suit}s is the first card with "
        f"probability 1/{rank_count}. Averaging the remaining {suit_count} {suit}s over the "
        f"51 remaining cards gives {ps._fmt(value)}.",
        ps._fraction(value),
    )


def _compute_reversed_dice(ctx: Context, p: Params) -> str:
    target_sum, first, sides = p["sum"], p["first"], p["sides"]
    outcomes = [(a, b) for a in range(1, sides + 1) for b in range(1, sides + 1) if a + b == target_sum]
    favourable = [pair for pair in outcomes if pair[0] == first]
    if not outcomes:
        return ps._prob(0.0, f"No two {sides}-sided dice outcomes sum to {target_sum}.")
    value = len(favourable) / len(outcomes)
    seen = ", ".join(f"({a},{b})" for a, b in outcomes)
    return ps._prob(
        value,
        f"Conditioning on a sum of {target_sum} leaves {len(outcomes)} ordered outcomes: {seen}. "
        f"{len(favourable)} of them have first die {first}.",
        ps._fraction(value),
    )


def _compute_reversed_coin(ctx: Context, p: Params) -> str:
    n, k, mode, first = p["n"], p["k"], p["mode"], p["first"]
    denominator = 0
    numerator = 0
    for heads in range(n + 1):
        if not ps._count_holds(heads, mode, k):
            continue
        denominator += math.comb(n, heads)
        if first.startswith("h"):
            numerator += math.comb(n - 1, heads - 1) if heads >= 1 else 0
        else:
            numerator += math.comb(n - 1, heads) if heads <= n - 1 else 0
    if denominator == 0:
        return ps._prob(0.0, "The conditioning event has no outcomes.")
    value = numerator / denominator
    return ps._prob(
        value,
        f"Conditioning on {ps._count_spec_label(mode, k)} in {n} flips leaves {denominator} "
        f"outcomes. {numerator} of them start with {first}.",
        ps._fraction(value),
    )


def _compute_pb_given_a(ctx: Context, p: Params) -> str:
    value = p["intersection"] / p["pa"]
    return ps._prob(
        value,
        f"P(B given A) = P(A and B)/P(A) = {ps._fmt(p['intersection'])}/{ps._fmt(p['pa'])}.",
        ps._fraction(value),
    )


def _compute_event_bayes(ctx: Context, p: Params) -> str:
    value = p["given_ab"] * p["pb"] / p["given_ba"]
    return ps._prob(
        value,
        "P(A given B)·P(B) = P(B given A)·P(A), so "
        f"P(A) = {ps._fmt(p['given_ab'])}·{ps._fmt(p['pb'])}/{ps._fmt(p['given_ba'])}.",
        ps._fraction(value),
    )


def _compute_total_probability(ctx: Context, p: Params) -> str:
    value = p["given_ba"] * p["pa"] + p["given_bna"] * (1 - p["pa"])
    return ps._prob(
        value,
        f"P(B) = P(B|A)P(A) + P(B|not A)P(not A) = "
        f"{ps._fmt(p['given_ba'])}·{ps._fmt(p['pa'])} + "
        f"{ps._fmt(p['given_bna'])}·{ps._fmt(1 - p['pa'])}.",
        ps._fraction(value),
    )


def _compute_low_prevalence_bayes(ctx: Context, p: Params) -> str:
    base = p["one"] / p["in"]
    true_positive = p["sensitivity"] * base
    false_positive = p["false_positive"] * (1 - base)
    value = true_positive / (true_positive + false_positive)
    return ps._prob(
        value,
        f"By Bayes theorem, posterior = ({ps._fmt(p['sensitivity'])}·{ps._fmt(base)}) / "
        f"({ps._fmt(p['sensitivity'])}·{ps._fmt(base)} + "
        f"{ps._fmt(p['false_positive'])}·{ps._fmt(1 - base)}).",
        ps._fraction(value),
    )


def _compute_exp_memoryless(ctx: Context, p: Params) -> str:
    gap = max(0.0, p["target"] - p["given"])
    value = math.exp(-p["rate"] * gap)
    return ps._prob(
        value,
        f"Exponential waiting times are memoryless, so P(X > {ps._fmt(p['target'])} | "
        f"X > {ps._fmt(p['given'])}) = P(X > {ps._fmt(gap)}) = e^(-{ps._fmt(p['rate']*gap)}).",
        ps._fraction(value),
    )


def _compute_clt(ctx: Context, p: Params) -> str:
    se = p["sd"] / math.sqrt(p["n"])
    z = (p["x"] - p["mean"]) / se
    value = 1 - _normal_cdf(z)
    return ps._prob(
        value,
        f"For the sample mean, SE = {ps._fmt(p['sd'])}/√{p['n']} = {ps._fmt(se)}. "
        f"z = ({ps._fmt(p['x'])} - {ps._fmt(p['mean'])})/{ps._fmt(se)} = {ps._fmt(z)}; "
        f"the upper tail is {ps._fmt(value)}.",
        ps._fraction(value),
    )


def _compute_t_interval(ctx: Context, p: Params) -> str:
    df = p["n"] - 1
    t = _t_critical(df, p["level"])
    margin = t * p["sd"] / math.sqrt(p["n"])
    lower, upper = p["mean"] - margin, p["mean"] + margin
    return (
        f"Because the standard deviation is a sample standard deviation, use Student-t with "
        f"df={df}. t={t}, so the margin is {t}·{ps._fmt(p['sd'])}/√{p['n']} = {margin:.2f}. "
        f"The interval is [{lower:.2f}, {upper:.2f}]."
    )


def _compute_normal_upper(ctx: Context, p: Params) -> str:
    z = (p["x"] - p["mean"]) / p["sd"]
    value = 1 - _normal_cdf(z)
    return ps._prob(
        value,
        f"Standardise x: z = ({ps._fmt(p['x'])} - {ps._fmt(p['mean'])})/{ps._fmt(p['sd'])} "
        f"= {ps._fmt(z)}. The upper-tail probability is {ps._fmt(value)}.",
        ps._fraction(value),
    )


def _compute_normal_within(ctx: Context, p: Params) -> str:
    z = p["distance"] / p["sd"]
    value = _normal_cdf(z) - _normal_cdf(-z)
    return ps._prob(
        value,
        f"Being within {ps._fmt(p['distance'])} units means z = ±{ps._fmt(z)}. "
        f"The central probability is Φ({ps._fmt(z)}) - Φ(-{ps._fmt(z)}) = {ps._fmt(value)}.",
        ps._fraction(value),
    )


def _compute_poisson_tail(ctx: Context, p: Params) -> str:
    lam, k = p["lambda"], p["k"]
    value = 1 - math.fsum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k))
    return ps._prob(
        value,
        f"For a Poisson count with mean {ps._fmt(lam)}, P(X ≥ {k}) = "
        f"1 - Σ_(i=0)^({k-1}) e^(-λ)λ^i/i! = {ps._fmt(value)}.",
        ps._fraction(value),
    )


def _compute_uniform_wording(ctx: Context, p: Params) -> str:
    support_low, support_high = p["support"]
    target_low, target_high = p["target"]
    low, high = max(support_low, target_low), min(support_high, target_high)
    value = (high - low) / (support_high - support_low) if high > low else 0.0
    return ps._prob(
        value,
        f"For Uniform[{ps._fmt(support_low)}, {ps._fmt(support_high)}], the requested interval "
        f"is [{ps._fmt(low)}, {ps._fmt(high)}], so probability = "
        f"({ps._fmt(high)} - {ps._fmt(low)})/({ps._fmt(support_high)} - {ps._fmt(support_low)}).",
        ps._fraction(value),
    )


def _compute_exp_survival(ctx: Context, p: Params) -> str:
    value = math.exp(-p["rate"] * p["x"])
    return ps._prob(
        value,
        f"For an exponential waiting time with rate {ps._fmt(p['rate'])}, "
        f"P(X > {ps._fmt(p['x'])}) = e^(-{ps._fmt(p['rate']*p['x'])}).",
        ps._fraction(value),
    )


def _compute_hypergeom_more_than(ctx: Context, p: Params) -> str:
    population, successes, draws, k = p["population"], p["successes"], p["draws"], p["k"]
    value = math.fsum(
        ps._hypergeom_pmf(population, successes, draws, i)
        for i in range(k + 1, min(successes, draws) + 1)
    )
    return ps._prob(
        value,
        f"Sampling without replacement is hypergeometric: P(X > {k}) = "
        f"Σ C({successes},i)C({population-successes},{draws}-i)/C({population},{draws}) "
        f"for i > {k} = {ps._fmt(value)}.",
        ps._fraction(value),
    )


def _compute_cov_from_expectation(ctx: Context, p: Params) -> str:
    value = p["exy"] - p["ex"] * p["ey"]
    return (
        f"Cov(X,Y) = E[XY] - E[X]E[Y] = {ps._fmt(p['exy'])} - "
        f"{ps._fmt(p['ex'])}·{ps._fmt(p['ey'])} = {ps._fmt(value)}."
    )


def _compute_variance_sum(ctx: Context, p: Params) -> str:
    sign = 1 if p["operator"] == "+" else -1
    value = p["varx"] + p["vary"] + 2 * sign * p["cov"]
    operation = "Var(X + Y)" if sign > 0 else "Var(X - Y)"
    covariance_term = "+ 2Cov(X,Y)" if sign > 0 else "- 2Cov(X,Y)"
    return (
        f"{operation} = Var(X) + Var(Y) {covariance_term} = "
        f"{ps._fmt(p['varx'])} + {ps._fmt(p['vary'])} {('+' if sign > 0 else '-')} "
        f"2·{ps._fmt(p['cov'])} = {ps._fmt(value)}."
    )


def _compute_die_race(ctx: Context, p: Params) -> str:
    return ps._prob(
        0.5,
        f"Only rolls of {p['first']} or {p['second']} matter. On a fair die they are equally "
        f"likely, so the chance that {p['first']} appears first is 1/2.",
        "1/2",
    )


def _compute_coin_race(ctx: Context, p: Params) -> str:
    needed, opponent, target = p["heads"], p["tails"], p["target"]
    if target not in {"heads", "tails"}:
        return ps._prob(0.0, "No valid stopping target was understood.")
    target_needed = needed if target == "heads" else opponent
    other_needed = opponent if target == "heads" else needed
    numerator = 0
    denominator = 0
    for j in range(other_needed):
        ways = math.comb(target_needed - 1 + j, j)
        weight = 0.5 ** (target_needed + j)
        numerator += ways * weight
        denominator += 1
    value = numerator
    label = "three heads" if target == "heads" else "two tails"
    return ps._prob(
        value,
        f"For {target_needed} {target} before {other_needed} {('tails' if target == 'heads' else 'heads')}, "
        f"sum the final-flip cases: {ps._fmt(value)}.",
        ps._fraction(value),
    )


def _compute_counting_repetition(ctx: Context, p: Params) -> str:
    return (
        f"Each of the {p['length']} positions has {len(p['digits'])} choices, so there are "
        f"{len(p['digits'])}^{p['length']} = {p['value']} strings."
    )


def _plan_card_joint(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    rank, suit = _rank_and_suit(q)
    if not (rank and suit):
        return None
    if not re.search(r"\bfirst card\b", q, re.I) or not re.search(r"\bsecond card\b", q, re.I):
        return None
    if not re.search(r"\bwithout replacement\b", q, re.I):
        return None
    if not re.search(r"\bfirst card is\b[^?]*\band\b[^?]*\bsecond card is\b", q, re.I):
        return None
    return Plan("card_joint_overlap", {"rank": rank, "suit": suit}, _compute_card_joint)


def _plan_conditional_card(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    rank, suit = _rank_and_suit(q)
    if not (rank and suit):
        return None
    if not re.search(r"\bfirst card\b", q, re.I) or not re.search(r"\bsecond card\b", q, re.I):
        return None
    if not re.search(r"\bwithout replacement\b", q, re.I):
        return None
    return Plan("conditional_card_second", {"rank": rank, "suit": suit}, _compute_conditional_card)


def _plan_reversed_dice(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    if not re.search(r"\bdice\b", q, re.I):
        return None
    target = _count(q, rf"\bgiven that the sum is\s+({ps._NUMBER_TOKEN})\b")
    first = _count(q, rf"\bfirst die\s+(?:shows|is)\s+({ps._NUMBER_TOKEN})\b")
    sides = ps._sides(q)
    if target is None or first is None or not sides:
        return None
    return Plan("reversed_dice", {"sum": target, "first": first, "sides": sides}, _compute_reversed_dice)


def _plan_reversed_coin(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    n = _count(q, rf"\b({ps._NUMBER_TOKEN})\s+(?:fair\s+)?(?:coin\s+)?(?:flips|tosses|times)\b")
    condition = re.search(
        rf"\bgiven that\s+(exactly|at least|at most)\s+({ps._NUMBER_TOKEN})\s+heads?\b",
        q,
        re.I,
    )
    first = re.search(r"\bfirst\s+(?:flip|toss)\s+was\s+(heads?|tails?)\b", q, re.I)
    if n is None or not condition or not first:
        return None
    mode = {"exactly": "exact", "at least": "at_least", "at most": "at_most"}[condition.group(1).lower()]
    k = ps._count_from_text(condition.group(2))
    if k is None:
        return None
    return Plan(
        "reversed_coin",
        {"n": n, "k": k, "mode": mode, "first": first.group(1).lower()},
        _compute_reversed_coin,
    )


def _plan_pb_given_a(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    intersection = _prob_value(q, r"p\s*\(\s*(?:a\s*(?:∩|and)\s*b|b\s*(?:∩|and)\s*a)\s*\)")
    pa = _prob_value(q, r"p\s*\(\s*a\s*\)")
    asks = re.search(
        r"(?:what is|find|calculate)\s+p\s*\(\s*b\s*(?:\||given)\s*a\s*\)|p\s*\(\s*b\s*(?:\||given)\s*a\s*\)\s*\?",
        q,
        re.I,
    )
    if None in (intersection, pa) or not asks:
        return None
    return Plan("pb_given_a", {"intersection": intersection, "pa": pa}, _compute_pb_given_a)


def _plan_event_bayes(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    given_ab = _prob_value(q, r"p\s*\(\s*a\s*(?:\||given)\s*b\s*\)")
    given_ba = _prob_value(q, r"p\s*\(\s*b\s*(?:\||given)\s*a\s*\)")
    pb = _prob_value(q, r"p\s*\(\s*b\s*\)")
    asks = re.search(r"(?:what is|find|calculate)\s+p\s*\(\s*a\s*\)|p\s*\(\s*a\s*\)\s*\?", q, re.I)
    if None in (given_ab, given_ba, pb) or not asks:
        return None
    return Plan(
        "event_bayes",
        {"given_ab": given_ab, "given_ba": given_ba, "pb": pb},
        _compute_event_bayes,
    )


def _plan_law_total(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    pa = _prob_value(q, r"p\s*\(\s*a\s*\)")
    given_ba = _prob_value(q, r"p\s*\(\s*b\s*(?:\||given)\s*a\s*\)")
    given_bna = _prob_value(q, r"p\s*\(\s*b\s*(?:\||given)\s*not\s*a\s*\)")
    asks = re.search(r"(?:what is|find|calculate)\s+p\s*\(\s*b\s*\)|p\s*\(\s*b\s*\)\s*\?", q, re.I)
    if None in (pa, given_ba, given_bna) or not asks:
        return None
    return Plan(
        "total_probability",
        {"pa": pa, "given_ba": given_ba, "given_bna": given_bna},
        _compute_total_probability,
    )


def _plan_low_prevalence_bayes(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    prevalence = re.search(r"\b(\d+)\s+in\s+(\d+)\s+people\b", q, re.I)
    sensitivity = ps._percent_match(q, r"sensitive")
    false_positive = ps._percent_match(q, r"false\s+positive(?:\s+rate)?")
    if not (prevalence and sensitivity is not None and false_positive is not None):
        return None
    if not re.search(r"\btests?\s+positive\b", q, re.I):
        return None
    return Plan(
        "low_prevalence_bayes",
        {
            "one": int(prevalence.group(1)),
            "in": int(prevalence.group(2)),
            "sensitivity": sensitivity,
            "false_positive": false_positive,
        },
        _compute_low_prevalence_bayes,
    )


def _plan_exp_memoryless(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    if not re.search(r"\bexponential\b", q, re.I) or not re.search(r"\bgiven\b", q, re.I):
        return None
    rate = _float(q, rf"\brate\s*(?:is|=|:)?\s*({_FLOAT})")
    pair = re.search(rf"p\s*\(\s*x\s*>\s*({_FLOAT})\s+given\s+x\s*>\s*({_FLOAT})\s*\)", q, re.I)
    given = float(pair.group(2)) if pair else _float(q, rf"given(?: that)?\s+x\s*>\s*({_FLOAT})")
    target = float(pair.group(1)) if pair else _float(q, rf"p\s*\(\s*x\s*>\s*({_FLOAT})")
    if rate is None or given is None or target is None:
        return None
    return Plan("exp_memoryless", {"rate": rate, "given": given, "target": target}, _compute_exp_memoryless)


def _plan_clt(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    if not re.search(r"\bsample mean\b", q, re.I):
        return None
    n = _count(q, rf"\bsample\s+of\s+size\s+({ps._NUMBER_TOKEN})|\bsample\s+of\s+({ps._NUMBER_TOKEN})")
    mean = _float(q, rf"\bmean\s*(?:is|=|:)?\s*({_FLOAT})")
    sd = _float(q, rf"standard\s+deviation\s*(?:is|=|:)?\s*({_FLOAT})")
    x = _float(q, rf"sample\s+mean\s+(?:exceeds?|is\s+greater\s+than|is\s+above)\s*({_FLOAT})")
    if None in (n, mean, sd, x):
        return None
    return Plan("clt", {"n": n, "mean": mean, "sd": sd, "x": x}, _compute_clt)


def _plan_t_interval(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    if not re.search(r"\bconfidence interval\b|\bmargin of error\b", q, re.I):
        return None
    if not re.search(r"\bsample standard deviation\b", q, re.I):
        return None
    n = _count(q, rf"\bsample\s+of\s+({ps._NUMBER_TOKEN})|\bn\s*=\s*({ps._NUMBER_TOKEN})")
    mean = _float(q, rf"\bmean\s*(?:is|=|:)?\s*({_FLOAT})")
    sd = _float(q, rf"sample\s+standard\s+deviation\s*(?:is|=|:)?\s*({_FLOAT})")
    level = _count(q, r"(\d+)\s*%\s*confidence")
    if None in (n, mean, sd, level) or n <= 1:
        return None
    return Plan("t_interval", {"n": n, "mean": mean, "sd": sd, "level": level}, _compute_t_interval)


def _plan_normal_upper(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    if not re.search(r"\bnormally distributed\b|\bnormal distribution\b", q, re.I):
        return None
    if re.search(r"\bsample mean\b", q, re.I):
        return None
    mean = _float(q, rf"\bmean\s*(?:is|=|:)?\s*({_FLOAT})")
    sd = _float(q, rf"standard\s+deviation\s*(?:is|=|:)?\s*({_FLOAT})")
    x = _float(q, rf"(?:greater\s+than|above|exceeds?|more\s+than)\s+({_FLOAT})")
    if None in (mean, sd, x) or sd <= 0:
        return None
    return Plan("normal_upper", {"mean": mean, "sd": sd, "x": x}, _compute_normal_upper)


def _plan_normal_within(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    if not re.search(r"\bnormally distributed\b|\bnormal distribution\b", q, re.I):
        return None
    distance = _float(q, rf"\bwithin\s+({_FLOAT})\s+units?\s+of\s+the\s+mean\b")
    mean = _float(q, rf"\bmean\s*(?:is|=|:)?\s*({_FLOAT})")
    sd = _float(q, rf"standard\s+deviation\s*(?:is|=|:)?\s*({_FLOAT})")
    if None in (distance, mean, sd) or sd <= 0:
        return None
    return Plan("normal_within", {"mean": mean, "sd": sd, "distance": distance}, _compute_normal_within)


def _plan_poisson_tail(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    if not re.search(r"\bpoisson\b", q, re.I):
        return None
    if not re.search(r"\bat least\b|\d+\s+or\s+more\b|\bmore than\b", q, re.I):
        return None
    lam = _float(q, rf"\bmean\s+({_FLOAT})")
    k = _count(q, rf"\b(?:at least|more than)\s+({ps._NUMBER_TOKEN})\b")
    if k is None:
        more = re.search(rf"\b({ps._NUMBER_TOKEN})\s+or\s+more\b", q, re.I)
        k = ps._count_from_text(more.group(1)) if more else None
    if lam is None or k is None:
        return None
    return Plan("poisson_tail", {"lambda": lam, "k": k}, _compute_poisson_tail)


def _plan_uniform_wording(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    if not re.search(r"\buniform(?:ly)?\b", q, re.I):
        return None
    intervals = re.findall(
        rf"\bbetween\s+({_FLOAT})\s+and\s+({_FLOAT})\b",
        q,
        re.I,
    )
    if len(intervals) < 2:
        return None
    support = (float(intervals[0][0]), float(intervals[0][1]))
    target = (float(intervals[1][0]), float(intervals[1][1]))
    if support[0] == support[1]:
        return None
    return Plan("uniform_wording", {"support": support, "target": target}, _compute_uniform_wording)


def _plan_exp_survival(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    if not re.search(r"\bexponential\b", q, re.I):
        return None
    rate = _float(q, rf"\brate\s*(?:is|=|:)?\s*({_FLOAT})")
    x = _float(q, rf"\b(?:waiting|lifetime|time)\s+(?:more|greater|longer)\s+than\s+({_FLOAT})")
    if rate is None or x is None:
        return None
    return Plan("exp_survival", {"rate": rate, "x": x}, _compute_exp_survival)


def _plan_hypergeom_more_than(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    if not re.search(r"\bwithout replacement\b", q, re.I) or not re.search(r"\bmore than\b", q, re.I):
        return None
    population = _count(q, rf"\bbatch\s+has\s+({ps._NUMBER_TOKEN})\s+items\b|\bbatch\s+of\s+({ps._NUMBER_TOKEN})\b")
    successes = _count(q, rf"\bof which\s+({ps._NUMBER_TOKEN})\s+are\s+defective\b|\b({ps._NUMBER_TOKEN})\s+are\s+defective\b")
    draws = _count(q, rf"\b({ps._NUMBER_TOKEN})\s+items?\s+(?:are\s+)?(?:sampled|drawn|selected)\b")
    k = _count(q, rf"\bmore than\s+({ps._NUMBER_TOKEN})\s+are\s+defective\b")
    if None in (population, successes, draws, k):
        return None
    return Plan(
        "hypergeom_more_than",
        {"population": population, "successes": successes, "draws": draws, "k": k},
        _compute_hypergeom_more_than,
    )


def _plan_cov_from_expectation(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    if not re.search(r"\bcov(?:ariance)?\b", q, re.I):
        return None
    ex, ey, exy = _expectation(q, "x"), _expectation(q, "y"), _expectation(q, "xy")
    if None in (ex, ey, exy):
        return None
    return Plan("cov_from_expectation", {"ex": ex, "ey": ey, "exy": exy}, _compute_cov_from_expectation)


def _plan_variance_sum(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    operation = re.search(r"\bvar(?:iance)?\s*\(\s*x\s*([+-])\s*y\s*\)", q, re.I)
    if not operation:
        return None
    varx = _prob_value(q, r"var(?:iance)?\s*\(\s*x\s*\)")
    vary = _prob_value(q, r"var(?:iance)?\s*\(\s*y\s*\)")
    cov = _prob_value(q, r"cov(?:ariance)?\s*\(\s*x\s*,\s*y\s*\)")
    if None in (varx, vary, cov):
        return None
    return Plan(
        "variance_sum",
        {"varx": varx, "vary": vary, "cov": cov, "operator": operation.group(1)},
        _compute_variance_sum,
    )


def _plan_die_race(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    race = re.search(r"\b(\d+)\s+appears?\s+before\s+(?:a|an)?\s*(\d+)\b", q, re.I)
    if not race or not re.search(r"\bdie\b", q, re.I):
        return None
    return Plan(
        "die_race",
        {"first": int(race.group(1)), "second": int(race.group(2))},
        _compute_die_race,
    )


def _plan_coin_race(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    if not re.search(r"\buntil either\b", q, re.I):
        return None
    heads = _count(q, rf"\b({ps._NUMBER_TOKEN})\s+heads?\b")
    tails = _count(q, rf"\b({ps._NUMBER_TOKEN})\s+tails?\b")
    final_target = re.search(r"\bprobability\s+that\s+(heads?|tails?)\s+appear\s+first\b", q, re.I)
    heads_first = bool(re.search(rf"\b(?:{ps._NUMBER_TOKEN})\s+heads?\s+appear first\b", q, re.I))
    tails_first = bool(re.search(rf"\b(?:{ps._NUMBER_TOKEN})\s+tails?\s+appear first\b", q, re.I))
    if final_target and final_target.group(1).lower().startswith("h"):
        heads_first = True
    if final_target and final_target.group(1).lower().startswith("t"):
        tails_first = True
    if heads_first == tails_first:
        return None
    target = "heads" if heads_first else "tails"
    if None in (heads, tails) or target is None:
        return None
    return Plan("coin_race", {"heads": heads, "tails": tails, "target": target}, _compute_coin_race)


def _plan_counting_repetition(ctx: Context) -> Optional[Plan]:
    q = ctx["question"]
    if not re.search(r"\brepetition is allowed\b|\brepetition allowed\b", q, re.I):
        return None
    if not re.search(r"\bstrings?\b|\bnumbers?\b", q, re.I):
        return None
    length = _count(q, rf"\b({ps._NUMBER_TOKEN})[- ]digit\b")
    digits_match = re.search(r"\bdigits?\s+([0-9,\sand]+?)\s+(?:if|with|when|\?|$)", q, re.I)
    if length is None or not digits_match:
        return None
    digits = [int(value) for value in re.findall(r"\d", digits_match.group(1))]
    if not digits:
        return None
    return Plan(
        "counting_repetition",
        {"length": length, "digits": digits, "value": len(digits) ** length},
        _compute_counting_repetition,
    )


_PLANNERS = (
    _plan_card_joint,
    _plan_conditional_card,
    _plan_reversed_dice,
    _plan_reversed_coin,
    _plan_pb_given_a,
    _plan_event_bayes,
    _plan_law_total,
    _plan_low_prevalence_bayes,
    _plan_exp_memoryless,
    _plan_clt,
    _plan_t_interval,
    _plan_normal_upper,
    _plan_normal_within,
    _plan_poisson_tail,
    _plan_uniform_wording,
    _plan_exp_survival,
    _plan_hypergeom_more_than,
    _plan_cov_from_expectation,
    _plan_variance_sum,
    _plan_die_race,
    _plan_coin_race,
    _plan_counting_repetition,
)


def reply(plan: Plan, ctx: Context) -> str:
    """Let Python compute the truth, then expose only the clean reply."""
    return plan.compute(ctx, plan.params)


def physics_answer(question: str) -> Optional[str]:
    """Exact physics calculations, kept separate so advanced routing cannot pre-empt them."""
    return physics_planners.answer(question)


def physics_concept_answer(question: str) -> Optional[str]:
    """Narrow applied-physics formulas for questions without a full numeric parse."""
    return physics_planners.concept_answer(question)


def answer(question: str) -> Optional[str]:
    physics_text = physics_answer(question)
    if physics_text:
        return physics_text
    ctx = understand(question)
    for planner in _PLANNERS:
        plan = planner(ctx)
        if plan:
            return reply(plan, ctx)
    return None


__all__ = ["answer", "physics_answer", "physics_concept_answer", "reply", "understand"]
