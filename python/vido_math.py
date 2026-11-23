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


def _parse_math(expression: str):
    return parse_expr(expression, local_dict=SYMBOLS, transformations=TRANSFORMS, evaluate=True)


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

    if re.search(r"\b(integrat(e|ion)|integral|antiderivative)\b|∫", q):
        match = re.search(
            r"(?:integrat(?:e|ion)|integral|antiderivative|∫)\s*(?:of\s*)?(.+?)"
            r"(?:\s+with\s+respect\s+to\s+([a-z]))?\s*$",
            q,
        )
        if match:
            try:
                expr = _parse_math(match.group(1))
                variable = SYMBOLS.get(match.group(2) or "x", Symbol("x"))
                return _sympy_answer("The integral of", expr, integrate(expr, variable))
            except Exception:
                return "I could not parse that function for integration."

    if re.search(r"\b(differentiat(e|ion)|derivative|differentiate)\b", q):
        match = re.search(r"(?:differentiat(?:e|ion)|derivative\s+of|differentiate)\s*(?:of\s*)?(.+?)\s*$", q)
        if match:
            try:
                expr = _parse_math(match.group(1))
                return _sympy_answer("The derivative of", expr, diff(expr, Symbol("x")))
            except Exception:
                return "I could not parse that function for differentiation."

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
