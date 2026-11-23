"""Math tools for ViDo.

Basic arithmetic uses the Shunting Yard algorithm so expressions such as
`2 * 9` or `(12 + 3) / 5` are evaluated without using eval(). Advanced queries
(integration, extrema, determinants, eigenvalues, ...) are routed to SymPy.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

from sympy import (
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

_INTEGRATION_PATTERN = re.compile(
    r"(?:integrat(?:e|ion|ing)|integral|antiderivative|∫)\s*(?:of\s*)?(.*?)"
    r"(?:\s+with\s+respect\s+to\s+([a-z]))?\s*$",
    re.I,
)
_DERIVATIVE_PATTERN = re.compile(
    r"(?:differentiat(?:e|ion|ing)|derivative\s+of|differentiate)\s*(?:of\s*)?(.*?)\s*$",
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


def handle_advanced(question: str) -> Optional[str]:
    """Return an answer for advanced math questions, or None if not applicable."""
    q = question.lower().strip()

    concept = _concept_explanation(q)
    if concept:
        return concept

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
            try:
                expr = _parse_math(match.group(1))
                return f"The solutions are {solve(expr, Symbol('x'))}."
            except Exception:
                return "I could not parse that equation."
    return None


def handle_query(question: str) -> Optional[str]:
    """Return a math answer when the question is mathematical, else None."""
    if not question or not question.strip():
        return None

    advanced = handle_advanced(question)
    if advanced:
        return advanced

    expression = _extract_expression(question)
    ok, value = evaluate_basic(expression)
    if ok:
        return f"{expression} = {_format_number(value)}."
    return None
