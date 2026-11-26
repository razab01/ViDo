"""Math tools for ViDo.

Basic arithmetic uses the Shunting Yard algorithm so expressions such as
`2 * 9` or `(12 + 3) / 5` are evaluated without using eval(). Advanced queries
(integration, extrema, determinants, eigenvalues, ...) are routed to SymPy.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import probability_stats

from sympy import (
    Eq,
    Matrix,
    N,
    Symbol,
    diff,
    expand,
    factor,
    integrate,
    limit,
    simplify,
    solve,
    sympify,
)
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)
SYMBOLS: Dict[str, Symbol] = {
    name: Symbol(name) for name in ("x", "y", "z", "t", "n", "a", "b", "c", "u", "v")
}

_FUNCTIONS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "log": math.log,
    "ln": math.log,
    "exp": math.exp,
    "abs": abs,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "%": 2, "^": 3, "u-": 4, "u+": 4}
_RIGHT_ASSOC = {"^", "u-", "u+"}

# Short, Wikipedia-style explanations for concept questions. These are used
# when the user asks *about* an operation ("how does integration work?") rather
# than giving a function to compute.
CONCEPT_EXPLANATIONS: Dict[str, str] = {
    "integration": (
        "Integration is a way of adding up many tiny pieces to find a total. On a graph it gives "
        "the area under a curve, and as an antiderivative it reverses differentiation: if F'(x) = f(x), "
        "then the integral of f(x) is F(x) + C. The “+ C” is the constant of integration because "
        "differentiating a constant gives zero."
    ),
    "differentiation": (
        "Differentiation finds the instantaneous rate of change of a function — the slope of its graph "
        "at a point. If integration combines tiny pieces into a total, differentiation splits a total "
        "into its tiny changes. For a power such as x^n, bring the exponent down in front and reduce "
        "the exponent by one."
    ),
    "calculus": (
        "Calculus is the branch of mathematics that studies continuous change. Differential calculus "
        "measures rates of change and slopes, while integral calculus adds up small pieces to find "
        "totals such as areas and accumulated quantities."
    ),
    "limit": (
        "A limit describes the value a function approaches as its input gets closer to some point. "
        "Limits are the foundation of calculus: they make the ideas of an instantaneous slope and an "
        "area under a curve precise."
    ),
}

STATISTICS_OVERVIEW = (
    "Probability measures how likely an event is, from 0 (impossible) to 1 (certain). Start with "
    "the sample space of all outcomes; when outcomes are equally likely, P(A) = favourable/total. "
    "Core rules: P(not A) = 1 - P(A); P(A or B) = P(A) + P(B) - P(A and B); for independent events "
    "P(A and B) = P(A)·P(B); and P(A given B) = P(A and B)/P(B). A random variable assigns numbers "
    "to outcomes, and its distribution gives their probabilities or density. Statistics uses that "
    "probability model to collect, summarise and infer from data: the mean and variance describe a "
    "sample (the mean is the expected value), while estimation and hypothesis tests generalise to a "
    "population. The law of large "
    "numbers and central limit theorem are the bridge between the two."
)

STATISTICS_TERMS: Dict[str, str] = {
    "mean": (
        "The mean is the arithmetic average: add all values and divide by how many there are. "
        "For a random variable, the equivalent long-run average is the expected value, "
        "E[X] = Σx·P(X=x)."
    ),
    "median": (
        "The median is the middle value after sorting the data. With an even number of values it is "
        "the average of the two middle ones. It resists extreme values better than the mean."
    ),
    "mode": (
        "The mode is the value that occurs most often in a data set. A set can have no mode, one "
        "mode or several modes, and the mode also works for categorical data."
    ),
    "range": (
        "The range is the largest value minus the smallest value. It is a quick measure of spread "
        "but is sensitive to outliers."
    ),
    "variance": (
        "Variance measures average squared spread around the mean: Var(X) = E[(X - μ)²]. "
        "Its square root is the standard deviation, which is in the original units."
    ),
    "standard deviation": (
        "Standard deviation is the square root of variance. It describes the typical distance of "
        "values from the mean in the same units as the data; a smaller value means tighter clustering."
    ),
    "expected value": (
        "Expected value is the long-run weighted average of a random variable: E[X] = Σx·P(X=x) "
        "for discrete outcomes, or ∫x·f(x)dx for a continuous density."
    ),
    "binomial distribution": (
        "A binomial distribution counts successes in n independent yes/no trials with the same "
        "success probability p: P(X=k) = C(n,k)·p^k·(1-p)^(n-k). Its mean is np and its variance "
        "is np(1-p)."
    ),
    "normal distribution": (
        "A normal distribution is a symmetric bell-shaped continuous distribution defined by its "
        "mean μ and standard deviation σ. About 68% of values lie within one σ of the mean and "
        "about 95% within two σ."
    ),
}

NULL_MATRIX_EIGENVALUES = (
    "All eigenvalues of a null (zero) matrix are 0. For an n×n zero matrix, 0 is the only "
    "eigenvalue, with algebraic multiplicity n, and every nonzero vector is an eigenvector."
)

NILPOTENT_EIGENVALUES = (
    "All eigenvalues of a nilpotent matrix are 0. If N^k = 0 and Nv = λv, then λ^k = 0, "
    "so every eigenvalue λ must be 0."
)

SKEW_ODD_DETERMINANT = (
    "The determinant of an odd-order skew-symmetric matrix is 0. Since Aᵀ = -A, "
    "det(A) = det(Aᵀ) = det(-A) = (-1)^n det(A); for odd n this gives det(A) = -det(A), hence det(A) = 0."
)

SKEW_EVEN_DETERMINANT = (
    "For an even-order skew-symmetric matrix the determinant is not forced to be 0. "
    "It equals the square of the Pfaffian, so a real example can have a positive determinant."
)

SKEW_GENERAL_DETERMINANT = (
    "For a skew-symmetric matrix A, det(A) = det(-A) = (-1)^n det(A). "
    "So an odd-order matrix always has determinant 0; an even-order matrix may have a non-zero determinant."
)

_INTEGRATION_PATTERN = re.compile(
    r"(?:integrat(?:e|ion|ing)|integral|antiderivative|∫)\s*(?:of\s*)?(.*?)"
    r"(?:\s+with\s+respect\s+to\s+([a-z]))?\s*$",
    re.I,
)
_DERIVATIVE_PATTERN = re.compile(
    r"(?:differentiat(?:e|ion|ing)|derivative\s+of|differentiate)\s*(?:of\s*)?(.*?)\s*$",
    re.I,
)

# Counting questions such as "how many 5 digit numbers can be formed from
# 1,2,3,4,5" are maths, not dataset lookups. Extract the requested length and
# the digit set, then use a permutation (or repetition) count.
_COUNT_QUESTION = re.compile(
    r"\bhow\s+many\s+(?:(?:even|odd)\s+)?(?:(\d+)\s*-?\s*digit\s+)?"
    r"(?:(?:even|odd)\s+)?(?:numbers?|ways?)\b",
    re.I,
)
_DIGIT_SOURCE = re.compile(
    r"\b(?:by|from|using|with|out\s+of|among)\s+(.+?)(?:[.?!]|$)",
    re.I,
)


def _tokenize(expression: str) -> List[str]:
    pattern = r"\d*\.?\d+|[A-Za-z_]\w*|[+\-*/^%()]"
    return re.findall(pattern, expression)


def _to_rpn(tokens: List[str]) -> List[str]:
    output: List[str] = []
    operators: List[str] = []
    expect_operand = True

    for token in tokens:
        if re.fullmatch(r"\d*\.?\d+", token) or token in _CONSTANTS:
            output.append(token)
            expect_operand = False
            continue
        if token in _FUNCTIONS:
            operators.append(token)
            expect_operand = True
            continue
        if token == "(":
            operators.append(token)
            expect_operand = True
            continue
        if token == ")":
            while operators and operators[-1] != "(":
                output.append(operators.pop())
            if operators and operators[-1] == "(":
                operators.pop()
            expect_operand = False
            continue

        current = token
        if token in ("+", "-") and expect_operand:
            current = "u+" if token == "+" else "u-"

        while operators:
            top = operators[-1]
            if top == "(" or top in _FUNCTIONS:
                break
            top_prec = _PRECEDENCE.get(top, 0)
            cur_prec = _PRECEDENCE.get(current, 0)
            if top_prec > cur_prec or (top_prec == cur_prec and current not in _RIGHT_ASSOC):
                output.append(operators.pop())
            else:
                break
        operators.append(current)
        expect_operand = current in _FUNCTIONS or current == "("

    while operators:
        output.append(operators.pop())
    return output


def _apply(operator: str, values: List[float]) -> float:
    if operator in ("u-", "u+"):
        value = values.pop()
        return -value if operator == "u-" else value
    right = values.pop()
    left = values.pop()
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator == "/":
        if right == 0:
            raise ZeroDivisionError("division by zero")
        return left / right
    if operator == "%":
        if right == 0:
            raise ZeroDivisionError("modulo by zero")
        return left % right
    if operator == "^":
        return left ** right
    raise ValueError(f"Unknown operator: {operator}")


def evaluate_basic(expression: str) -> Tuple[bool, float]:
    """Evaluate an arithmetic expression using Shunting Yard, never eval()."""
    tokens = _tokenize(expression)
    if not tokens:
        return False, 0.0
    try:
        rpn = _to_rpn(tokens)
        values: List[float] = []
        for token in rpn:
            if re.fullmatch(r"\d*\.?\d+", token):
                values.append(float(token))
            elif token in _CONSTANTS:
                values.append(_CONSTANTS[token])
            elif token in _FUNCTIONS:
                values.append(_FUNCTIONS[token](values.pop()))
            else:
                values.append(_apply(token, values))
        if len(values) != 1:
            return False, 0.0
        result = values[0]
        if isinstance(result, complex):
            return False, 0.0
        return True, result
    except (IndexError, KeyError, ValueError, ZeroDivisionError, OverflowError):
        return False, 0.0


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-10:
        return str(int(round(value)))
    return f"{value:.10g}"


def _extract_expression(question: str) -> str:
    """Take the arithmetic-looking slice from a natural question."""
    low = question.lower()
    low = re.sub(r"\b(what is|whats|what's|calculate|compute|solve)\b", " ", low)
    # Equations and reaction arrows are solver syntax, not flat arithmetic.
    # Without this guard, "2H2 + O2 -> 2H2O" can look like "2 + 2" to the
    # Shunting Yard extractor.
    if "=" in low or re.search(r"->|=>|\byields?\b", low):
        return ""
    # "20 m/s for 10 s" must reach the physics solver, not become 20 / 10.
    applied_science = re.search(
        r"m\s*/\s*s(?:\^?2)?|\b(?:kg|kilograms?|n|newtons?|j|joules?|w|watts?|"
        r"ohms?|volts?|amps?|mol|moles?|grams?|kelvin|atm|ph)\b|"
        r"\b(?:accelerat\w*|velocity|speed|distance|displacement|force|pressure|"
        r"current|resistance|work done|kinetic energy|potential energy)\b",
        low,
        re.I,
    )
    if applied_science:
        return ""
    # Avoid treating dataset dates as arithmetic.
    if re.search(r"\b20\d{2}-\d{2}-\d{2}\b", low):
        return ""
    candidate = re.sub(r"[^0-9.+\-*/^()% ]", " ", low)
    candidate = re.sub(r"\s+", " ", candidate).strip()
    if not re.search(r"[+\-*/^%]", candidate):
        return ""
    return candidate


def _normalize_spoken_math(text: str) -> str:
    """Translate a few spoken math phrases so 'x squared' becomes x^2."""
    value = str(text or "").strip()
    value = value.replace("×", "*").replace("÷", "/").replace("–", "-")
    value = re.sub(r"\bsquared\b", "^2", value, flags=re.I)
    value = re.sub(r"\bcubed\b", "^3", value, flags=re.I)
    value = re.sub(r"\btimes\b", "*", value, flags=re.I)
    value = re.sub(r"\bplus\b", "+", value, flags=re.I)
    value = re.sub(r"\bminus\b", "-", value, flags=re.I)
    return value.strip()


def _parse_math(expression: str):
    return parse_expr(
        _normalize_spoken_math(expression),
        local_dict=SYMBOLS,
        transformations=TRANSFORMS,
        evaluate=True,
    )


def _looks_like_expression(candidate: str) -> bool:
    """Reject prose such as 'happen' before implicit multiplication parses it."""
    text = _normalize_spoken_math(candidate).lower().strip(" .?!,")
    if not text:
        return False
    if re.search(r"\d|[+\-*/^=()]", text):
        return True
    if re.search(r"\b(?:sin|cos|tan|sqrt|log|ln|exp|abs)\b", text):
        return True
    return bool(re.search(r"\b[xytuvn]\b", text))


def _concept_explanation(q: str) -> Optional[str]:
    """Explain a math idea when no actual expression was supplied."""
    if re.search(r"\bcalculus\b", q) and not re.search(r"\d|[+\-*/^]", q):
        return CONCEPT_EXPLANATIONS["calculus"]

    match = _INTEGRATION_PATTERN.search(q)
    if match and not _looks_like_expression(match.group(1)):
        return CONCEPT_EXPLANATIONS["integration"]

    match = _DERIVATIVE_PATTERN.search(q)
    if match and not _looks_like_expression(match.group(1)):
        return CONCEPT_EXPLANATIONS["differentiation"]

    if re.search(r"\blimit\b", q) and not re.search(r"\blimit\s+of\s+.+\s+as\s+", q):
        return CONCEPT_EXPLANATIONS["limit"]
    return None


def _power_rule_hint(source: str) -> str:
    if re.search(r"\d|\^|\*\*|squared|cubed", str(source or ""), re.I):
        return (
            "For a power, raise the exponent by one and divide by the new exponent. "
        )
    return ""


def _integration_answer(expression, variable, value, source: str) -> str:
    result = f"The integral of {expression} with respect to {variable} is {value} + C."
    explanation = (
        "Integration is the reverse of differentiation: it adds up tiny pieces to find a total, "
        "such as the area under a curve. "
        + _power_rule_hint(source)
        + "The “+ C” is the constant of integration because a constant disappears when you differentiate."
    )
    return f"{result}\n\n{explanation}"


def _derivative_answer(expression, value, source: str) -> str:
    result = f"The derivative of {expression} is {value}."
    explanation = (
        "Differentiation measures the instantaneous rate of change — the slope of the curve at a point. "
        + _power_rule_hint(source).replace("raise the exponent by one and divide by the new exponent",
                                          "bring the exponent down and reduce it by one")
        + "That gives a formula for how quickly the function changes as its input changes."
    )
    return f"{result}\n\n{explanation}"


def _matrix_text(question: str) -> Optional[str]:
    """Accept [[1,2],[3,4]] and also [1,2],[3,4] without outer brackets."""
    nested = re.search(r"\[\[.*\]\]", question)
    if nested:
        return nested.group(0)
    rows = re.search(r"\[[^\]]+\](?:\s*,\s*\[[^\]]+\])+", question)
    if rows:
        return "[" + rows.group(0) + "]"
    return None


def _parse_matrix(text: str):
    inner = text.strip()[1:-1]
    rows = []
    for row_text in re.findall(r"\[([^\]]+)\]", inner):
        # Accept both [[1,2],[3,4]] and [[1 2 3],[4 5 6]]: whitespace inside a
        # row is treated the same as a comma.
        values: List[str] = []
        for chunk in row_text.split(","):
            values.extend(part for part in re.split(r"\s+", chunk.strip()) if part)
        rows.append([_parse_math(value) for value in values])
    return Matrix(rows)


def _sympy_answer(label: str, expression, value) -> str:
    rendered = N(value, 8) if hasattr(value, "free_symbols") is False else value
    return f"{label} {expression} = {rendered}."


def _named_matrix_answer(q: str) -> Optional[str]:
    """Answer questions about named matrices without requiring literal entries."""
    if re.search(r"\bnilpotent\b", q) and re.search(r"\beigen\s*-?\s*values?\b|\beigenvalues?\b", q):
        return NILPOTENT_EIGENVALUES

    if re.search(r"\b(?:skew[\s-]*symmetric|antisymmetric)\b", q) and re.search(
        r"\b(?:determinant|det)\b", q
    ):
        if re.search(r"\bodd\b", q):
            return SKEW_ODD_DETERMINANT
        if re.search(r"\beven\b", q):
            return SKEW_EVEN_DETERMINANT
        return SKEW_GENERAL_DETERMINANT

    if re.search(r"\b(?:null|zero)\s+matrix\b", q):
        if re.search(r"\beigen\s*-?\s*vectors?\b|\beigenvectors?\b", q):
            return (
                "For a null (zero) matrix, every nonzero vector is an eigenvector, because "
                "A·v = 0 = 0·v for every vector v. The only eigenvalue is 0."
            )
        if re.search(r"\beigen\s*-?\s*values?\b|\beigenvalues?\b", q):
            return NULL_MATRIX_EIGENVALUES
    return None


def _normalize_math_question(question: str) -> str:
    """Fix the common 'dd shape' typo before semantic math routing."""
    value = str(question or "").strip()
    value = re.sub(r"\bdd\s+(shape|order|sized?)\b", "odd \\1", value, flags=re.I)
    value = re.sub(r"\bdd\s+matrix\b", "odd matrix", value, flags=re.I)
    return value


def _statistics_answer(question: str) -> Optional[str]:
    """Give a complete definition when someone asks about a statistics idea."""
    value = str(question or "").strip()
    low = value.lower()
    if not re.search(
        r"\b(?:probability|statistics|mean|median|mode|variance|standard deviation|"
        r"expected value|binomial distribution|normal distribution)\b",
        low,
    ):
        return None
    if re.search(r"\b(?:difference|versus|vs\.?|compare)\b", low) and re.search(
        r"\bprobability\b", low
    ) and re.search(r"\bstatistics\b", low):
        return STATISTICS_OVERVIEW
    definition = re.match(
        r"^(?:what\s+is|what's|define|meaning\s+of|explain|tell\s+me\s+about)\s+"
        r"(?:the\s+)?(.+?)[.?!]*$",
        low,
    )
    if not definition:
        return None
    topic = definition.group(1).strip()
    if re.search(r"\bprobability\s+and\s+statistics\b|\bstatistics\s+and\s+probability\b", topic):
        return STATISTICS_OVERVIEW
    if topic in ("probability", "probability theory", "statistics", "statistical inference"):
        return STATISTICS_OVERVIEW
    for key, text in STATISTICS_TERMS.items():
        if topic == key:
            return text
    return None


def _split_combined_questions(question: str) -> List[str]:
    """Split 'X and also tell me Y' into independent math questions."""
    value = _normalize_math_question(question)
    parts = re.split(
        r"\s+\band\s+(?:also\s+)?(?:tell\s+me\s+)?|\s+\balso\s+tell\s+me\s+",
        value,
        flags=re.I,
    )
    return [part.strip(" .?!") for part in parts if part.strip(" .?!")]


def _falling_factors(available: int, take: int) -> List[int]:
    """Ordered choices for `take` positions without repetition."""
    if take < 0 or take > available:
        return []
    return list(range(available, available - take, -1))


def _product(values: List[int]) -> int:
    total = 1
    for value in values:
        total *= value
    return total


def _probability_text(numerator: int, denominator: int, explanation: str) -> str:
    divisor = math.gcd(numerator, denominator) or 1
    reduced_denominator = denominator // divisor
    fraction = (
        str(numerator // divisor)
        if reduced_denominator == 1
        else f"{numerator // divisor}/{reduced_denominator}"
    )
    value = numerator / denominator
    return f"{fraction} ({value:.10g}, or {value * 100:.10g}%). {explanation}"


_NUMBER_WORDS: Dict[str, int] = {
    "zero": 0, "a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "twice": 2,
    "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_NUMBER_PATTERN = (
    r"\d+|zero|one|single|two|twice|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
)


def _as_count(raw: str) -> Optional[int]:
    value = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw.lower())
    return value if value is not None and value >= 0 else None


def _probability_answer(question: str) -> Optional[str]:
    """Answer fair-coin and fair-die probability questions with explicit wording."""
    q = str(question or "").lower()
    if not re.search(r"\b(?:probability|chance|likelihood)\b", q):
        return None

    # A head/tail question is a coin question even when the word "coin" is
    # implied by "toss", and vice versa. Bias is only assumed when stated.
    coin = bool(
        re.search(r"\bcoin\b", q)
        or re.search(r"\bheads?\b|\btails?\b", q)
    )
    die = bool(re.search(r"\b(?:die|dice)\b", q))
    if not coin and not die:
        return None
    if re.search(r"\b(?:biased|unfair|loaded)\b", q):
        return "That depends on the bias. For a fair coin or die, each outcome is equally likely."

    if coin:
        count = 1
        match = re.search(
            rf"\b({_NUMBER_PATTERN})\s+(?:fair\s+)?(?:coin\s+)?"
            r"(?:times|tosses|flips|time|toss|flip)\b",
            q,
        )
        if match:
            parsed = _as_count(match.group(1))
            if parsed is None:
                return None
            count = parsed
        elif re.search(r"\btwice\b", q):
            count = 2
        if count <= 0:
            return None
        total = 2 ** count
        toss_word = "toss" if count == 1 else "tosses"

        def range_count(lower: int, upper: int) -> int:
            return sum(
                math.comb(count, k) for k in range(max(0, lower), min(count, upper) + 1)
            )

        def singular(raw: str) -> str:
            return "head" if raw.lower().startswith("head") else "tail"

        def plural(word: str, amount: int) -> str:
            return word if amount == 1 else word + "s"

        at_least = re.search(
            rf"\bat\s+least\s+({_NUMBER_PATTERN})\s+(heads?|tails?)\b", q
        )
        if at_least:
            wanted = _as_count(at_least.group(1))
            target = singular(at_least.group(2))
            if wanted is None:
                return None
            numerator = range_count(wanted, count)
            explanation = (
                f"In {count} fair {toss_word}, there are {total} equally likely outcomes; "
                f"{numerator} {'is' if numerator == 1 else 'are'} at least {wanted} "
                f"{plural(target, wanted)}."
            )
            return _probability_text(numerator, total, explanation)

        at_most = re.search(
            rf"\bat\s+most\s+({_NUMBER_PATTERN})\s+(heads?|tails?)\b", q
        )
        if at_most:
            wanted = _as_count(at_most.group(1))
            target = singular(at_most.group(2))
            if wanted is None:
                return None
            numerator = range_count(0, wanted)
            explanation = (
                f"In {count} fair {toss_word}, there are {total} equally likely outcomes; "
                f"{numerator} {'is' if numerator == 1 else 'are'} at most {wanted} "
                f"{plural(target, wanted)}."
            )
            return _probability_text(numerator, total, explanation)

        no_match = re.search(r"\b(?:no|none)\s+(heads?|tails?)\b", q)
        if no_match:
            target = singular(no_match.group(1))
            explanation = (
                f"In {count} fair {toss_word}, there are {total} equally likely outcomes; "
                f"only 1 has no {target}."
            )
            return _probability_text(1, total, explanation)

        all_match = re.search(r"\b(?:all|both)\s+(heads?|tails?)\b", q)
        if all_match:
            target = singular(all_match.group(1))
            explanation = (
                f"In {count} fair {toss_word}, there are {total} equally likely outcomes; "
                f"only 1 is all {target}."
            )
            return _probability_text(1, total, explanation)

        # "exactly single head", "a exactly one head", "exactly 2 heads" and
        # the shorter "two heads" all mean the binomial point probability.
        exact = re.search(
            rf"\b(?:exactly|precisely|just|only)?\s*(?:a\s+|an\s+)?"
            rf"({_NUMBER_PATTERN})\s+(heads?|tails?)\b",
            q,
        )
        if exact:
            wanted = _as_count(exact.group(1))
            target = singular(exact.group(2))
            if wanted is None:
                return None
            numerator = math.comb(count, wanted) if wanted <= count else 0
            explanation = (
                f"In {count} fair {toss_word}, there are {total} equally likely outcomes; "
                f"{numerator} {'is' if numerator == 1 else 'are'} exactly {wanted} "
                f"{plural(target, wanted)}."
            )
            return _probability_text(numerator, total, explanation)

        target = "head" if re.search(r"\bheads?\b", q) else (
            "tail" if re.search(r"\btails?\b", q) else None
        )
        if not target:
            return None
        if count == 1:
            explanation = (
                f"A fair coin has 2 equally likely outcomes, head and tail, so "
                f"P({target}) = 1/2."
            )
            return _probability_text(1, 2, explanation)
        one = _probability_text(
            math.comb(count, 1),
            total,
            f"exactly one {target} in {count} fair {toss_word}",
        )
        least_one = _probability_text(
            total - 1,
            total,
            f"at least one {target} in {count} fair {toss_word}",
        )
        return f"Specify how many {target}s you mean. {one} {least_one}"

    # Multi-dice questions need the joint distribution, not a single-face rule.
    if re.search(r"\b(?:\d+|two|three|four|five|six|seven|eight|nine|ten)\s+(?:fair\s+)?dice\b", q):
        return None

    face_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    }
    face_match = re.search(
        r"\b(?:getting|rolling|roll|show(?:s|ing)?)\s+(?:a\s+)?"
        r"(one|two|three|four|five|six|[1-6])\b",
        q,
    )
    if face_match:
        raw = face_match.group(1)
        value = int(raw) if raw.isdigit() else face_words[raw]
        if 1 <= value <= 6:
            explanation = f"A fair die has 6 equally likely faces, so P({raw}) = 1/6."
            return _probability_text(1, 6, explanation)
    if re.search(r"\beven\b", q) or re.search(r"\bodd\b", q):
        parity = "even" if re.search(r"\beven\b", q) else "odd"
        explanation = (
            f"A fair die has 3 {parity} faces out of 6, so P({parity}) = 1/2."
        )
        return _probability_text(3, 6, explanation)
    return None


def _counting_answer(question: str) -> Optional[str]:
    """Count digit arrangements before a dataset graph can claim the question."""
    q = str(question or "").lower()
    count_match = _COUNT_QUESTION.search(q)
    if not count_match:
        return None
    if not re.search(r"\b(?:form(?:ed)?|make|made|arrange|arranged|construct|created?)\b", q):
        return None

    source_match = _DIGIT_SOURCE.search(q)
    if not source_match:
        return None
    digits = list(dict.fromkeys(int(digit) for digit in re.findall(r"\b\d\b", source_match.group(1))))
    if not digits:
        return None

    available = len(digits)
    requested = int(count_match.group(1)) if count_match.group(1) else available
    if requested <= 0:
        return None

    with_repeat = bool(re.search(
        r"\b(?:with|allow(?:ing|ed)?|permit(?:ting|ted)?)\s+(?:repetition|repeats?|repeating)\b"
        r"|\bdigits?\s+(?:can|may)\s+be\s+(?:repeated|reused|re-used)\b",
        q,
        re.I,
    ))
    without_repeat = bool(re.search(
        r"\b(?:without|no)\s+(?:repetition|repeats?|repeating)\b"
        r"|\bdigits?\s+(?:cannot|can't|must\s+not)\s+be\s+(?:repeated|reused|re-used)\b"
        r"|\beach\s+digit\s+(?:used|appears)\s+(?:once|only\s+once)\b",
        q,
        re.I,
    ))
    repeat = with_repeat and not without_repeat

    # A multi-digit number cannot start with zero, so the first place loses one
    # choice whenever zero is in the supplied set.
    zero_blocks_first = 0 in digits
    first_choices = available - (1 if zero_blocks_first else 0)
    digit_text = ", ".join(str(digit) for digit in digits)
    if count_match.group(1):
        setup = f"For {requested}-digit numbers from {digit_text}"
    else:
        setup = f"Using all {available} digits {digit_text}"

    parity = None
    if re.search(r"\beven\b", q) and not re.search(r"\bodd\b", q):
        parity = "even"
    elif re.search(r"\bodd\b", q) and not re.search(r"\beven\b", q):
        parity = "odd"

    if not parity:
        if repeat:
            factors = [first_choices] + [available] * (requested - 1)
        else:
            factors = [first_choices] + _falling_factors(available - 1, requested - 1)

        if factors:
            total = _product(factors)
            shown = " × ".join(str(factor) for factor in factors)
            wording = "digits may repeat" if repeat else "digits are used at most once"
            answer = f"{total}. {setup}, {wording}: {shown} = {total}."
        else:
            answer = (
                f"0. {setup}, digits are used at most once, so a {requested}-digit "
                "number is not possible."
            )

        if zero_blocks_first:
            answer += (
                f" A number cannot start with 0, so the first place has "
                f"{first_choices} choices."
            )
        if not repeat:
            repeat_factors = [first_choices] + [available] * (requested - 1)
            answer += (
                f" If repetition were allowed, it would be {_product(repeat_factors)}."
            )
        return answer

    parity_digits = [digit for digit in digits if digit % 2 == (0 if parity == "even" else 1)]
    parity_text = ", ".join(str(digit) for digit in parity_digits)
    if not parity_digits:
        return f"0. {setup}, there is no {parity} digit in the supplied set."

    if repeat:
        if requested == 1:
            factors = [len(parity_digits)]
        else:
            factors = (
                [first_choices]
                + [available] * (requested - 2)
                + [len(parity_digits)]
            )
        total = _product(factors)
        shown = " × ".join(str(factor) for factor in factors)
        return (
            f"{total}. {setup}, the last digit must be {parity} ({parity_text}) "
            f"and digits may repeat: {shown} = {total}."
        )

    if requested > available:
        return (
            f"0. {setup}, digits are used at most once, so a {requested}-digit "
            "number is not possible."
        )
    if requested == 1:
        factors = [len(parity_digits)]
        total = _product(factors)
        shown = " × ".join(str(factor) for factor in factors)
        return (
            f"{total}. A 1-digit {parity} number from {digit_text}: "
            f"{shown} = {total}."
        )

    middle_factors = _falling_factors(available - 2, requested - 2)
    if not middle_factors and requested > 2:
        return f"0. {setup}, a {requested}-digit number is not possible."
    middle_factors = middle_factors or [1]
    last_nonzero = [digit for digit in parity_digits if digit != 0]
    terms: List[List[int]] = []
    if 0 in parity_digits:
        # Choosing 0 for the last place still leaves every nonzero digit for first.
        terms.append([1, available - 1] + middle_factors)
    if last_nonzero:
        first_after_nonzero = available - 1 - (1 if zero_blocks_first else 0)
        terms.append([len(last_nonzero), first_after_nonzero] + middle_factors)

    totals = [_product(term) for term in terms]
    total = sum(totals)
    shown = " + ".join(" × ".join(str(factor) for factor in term) for term in terms)
    if len(totals) == 1:
        calculation = f"{shown} = {total}"
    else:
        calculation = f"{shown} = {' + '.join(str(value) for value in totals)} = {total}"
    answer = (
        f"{total}. {setup}, the last digit must be {parity} ({parity_text}) "
        f"and digits are used at most once: {calculation}."
    )
    if zero_blocks_first:
        answer += " A number cannot start with 0, so the first place excludes 0."
    repeat_factors = (
        [len(parity_digits)]
        if requested == 1
        else [first_choices] + [available] * (requested - 2) + [len(parity_digits)]
    )
    answer += f" If repetition were allowed, it would be {_product(repeat_factors)}."
    return answer


def handle_advanced(question: str) -> Optional[str]:
    """Return an answer for advanced math questions, or None if not applicable."""
    q = question.lower().strip()

    concept = _concept_explanation(q)
    if concept:
        return concept

    # Named matrices such as null, nilpotent and skew-symmetric are common
    # questions; the user should not have to type a literal matrix.
    named = _named_matrix_answer(q)
    if named:
        return named

    matrix_text = _matrix_text(q)
    if re.search(r"\b(determinant|det)\b", q) and matrix_text:
        try:
            matrix = _parse_matrix(matrix_text)
            return f"The determinant is {simplify(matrix.det())}."
        except Exception:
            return "I could not read that matrix. Try something like determinant of [[1,2],[3,4]]."

    if re.search(r"\beigen\s*-?\s*vectors?\b|\beigenvectors?\b", q) and matrix_text:
        try:
            matrix = _parse_matrix(matrix_text)
            parts = []
            for value, _multiplicity, vectors in matrix.eigenvects():
                for vector in vectors:
                    entries = ", ".join(str(simplify(entry)) for entry in vector)
                    parts.append(f"({entries}) for eigenvalue {value}")
            return f"The eigenvectors are {'; '.join(parts)}."
        except Exception:
            return "I could not read that matrix. Try something like eigen vectors of [[1,2],[3,4]]."

    if re.search(r"\beigen\s*-?\s*values?\b|\beigenvalues?\b", q) and matrix_text:
        try:
            matrix = _parse_matrix(matrix_text)
            values = matrix.eigenvals()
            return f"The eigenvalues are {', '.join(str(v) for v in values.keys())}."
        except Exception:
            return "I could not read that matrix. Try something like eigen values of [[1,2],[3,4]]."

    match = _INTEGRATION_PATTERN.search(q)
    if match:
        source = match.group(1).strip()
        try:
            expr = _parse_math(source)
            variable = SYMBOLS.get(match.group(2) or "x", Symbol("x"))
            return _integration_answer(expr, variable, integrate(expr, variable), source)
        except Exception:
            return (
                CONCEPT_EXPLANATIONS["integration"]
                + "\n\nI could not read that exact function. Try something like “integrate x^2”."
            )

    match = _DERIVATIVE_PATTERN.search(q)
    if match:
        source = match.group(1).strip()
        try:
            expr = _parse_math(source)
            return _derivative_answer(expr, diff(expr, Symbol("x")), source)
        except Exception:
            return (
                CONCEPT_EXPLANATIONS["differentiation"]
                + "\n\nI could not read that exact function. Try something like “differentiate x^2”."
            )

    if re.search(r"\b(maxima|maximum|minima|minimum|extrema|turning point)\b", q):
        match = re.search(r"(?:maxim\w*|minim\w*|extrema|turning point)\s+(?:of\s+|for\s+)?(.+?)\s*$", q)
        if match:
            try:
                expr = _parse_math(match.group(1))
                derivative = diff(expr, Symbol("x"))
                stationary = solve(derivative, Symbol("x"))
                if not stationary:
                    return f"The expression {expr} has no real stationary point."
                points = []
                for point in stationary:
                    if point.is_real:
                        points.append(f"x = {point}, value = {simplify(expr.subs(Symbol('x'), point))}")
                want_max = "max" in q
                if len(points) == 1:
                    kind = "maximum" if want_max else "minimum"
                    return f"The {kind} is {points[0]}."
                return f"The stationary points are {'; '.join(points)}."
            except Exception:
                return "I could not parse that function for extrema."

    if re.search(r"\blimit\b", q):
        match = re.search(r"limit\s+of\s+(.+?)\s+as\s+([a-z])\s*(?:approaches|tends to|->)\s*(.+?)\s*$", q)
        if match:
            try:
                expr = _parse_math(match.group(1))
                variable = SYMBOLS.get(match.group(2), Symbol("x"))
                target = _parse_math(match.group(3))
                return _sympy_answer(f"The limit of {expr} as {variable} approaches {target} is", "", limit(expr, variable, target)).replace(" = ", " ")
            except Exception:
                return "I could not parse that limit."

    if re.search(r"\bsimplify\b", q):
        match = re.search(r"simplify\s+(.+?)\s*$", q)
        if match:
            try:
                return f"Simplified, that is {simplify(_parse_math(match.group(1)))}."
            except Exception:
                return "I could not parse that expression."

    if re.search(r"\bfactor\b", q):
        match = re.search(r"factor\s+(.+?)\s*$", q)
        if match:
            try:
                return f"Factored, that is {factor(_parse_math(match.group(1)))}."
            except Exception:
                return "I could not parse that expression."

    if re.search(r"\bexpand\b", q):
        match = re.search(r"expand\s+(.+?)\s*$", q)
        if match:
            try:
                return f"Expanded, that is {expand(_parse_math(match.group(1)))}."
            except Exception:
                return "I could not parse that expression."

    if re.search(r"\b(solve|roots?)\b", q):
        match = re.search(r"(?:solve|roots?\s+of)\s+(.+?)\s*$", q)
        if match:
            source = match.group(1).strip()
            try:
                if "=" in source:
                    left_text, right_text = source.split("=", 1)
                    equation = Eq(_parse_math(left_text), _parse_math(right_text))
                    variable = next(
                        iter(sorted(equation.free_symbols, key=lambda symbol: symbol.name)),
                        Symbol("x"),
                    )
                    solutions = solve(equation, variable)

                    def render_solution(value):
                        try:
                            return _format_number(float(N(value, 12)))
                        except Exception:
                            return str(value)

                    rendered = ", ".join(render_solution(value) for value in solutions)
                    return f"Solving {equation.lhs} = {equation.rhs} gives {variable} = {rendered}."
                expr = _parse_math(source)
                return f"The solutions are {solve(expr, Symbol('x'))}."
            except Exception:
                return "I could not parse that equation."
    return None


def handle_query(question: str) -> Optional[str]:
    """Return a math answer when the question is mathematical, else None."""
    if not question or not question.strip():
        return None

    parts = _split_combined_questions(question)
    if len(parts) > 1:
        answers: List[str] = []
        for part in parts:
            answer = handle_query(part)
            if not answer:
                break
            answers.append(answer)
        if len(answers) == len(parts):
            return "\n\n".join(answers)

    value = _normalize_math_question(question)
    # Conditional, stopping-time and comparison questions must reach the broad
    # numeric solvers before the short coin/die helper can answer a simpler
    # sub-question such as "at least one head" instead of the actual target.
    hard = probability_stats.hard_answer(value)
    if hard:
        return hard

    probability = _probability_answer(value)
    if probability:
        return probability

    counting = _counting_answer(value)
    if counting:
        return counting

    statistics = _statistics_answer(value)
    if statistics:
        return statistics

    # Broad probability/statistics engine: event algebra, distributions,
    # inference, sampling, counting and RAG-grounded concept explanations.
    broad = probability_stats.answer(value)
    if broad:
        return broad

    advanced = handle_advanced(value)
    if advanced:
        return advanced

    expression = _extract_expression(value)
    ok, number = evaluate_basic(expression)
    if ok:
        return f"{expression} = {_format_number(number)}."
    return None
