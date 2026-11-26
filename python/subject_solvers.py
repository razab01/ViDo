"""Applied solvers for ViDo's Python-first subject routing.

Phase 1 answers definition questions from subject cards. Phase 2 adds numerical
solvers for the three subjects where a calculation is expected: Mathematics,
Physics and Chemistry. The solvers are intentionally small and deterministic so
they keep working offline, and they only run after a question has been routed to
one of those subjects (or carries unmistakable units/formulas for one of them).

Answers never include engine or source badges. Wikipedia evidence is checked by
the evaluation harness, not by the answer text.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from sympy import Eq, N, expand, factor, simplify, solve as sympy_solve

from subject_rag import Classification
from vido_math import _format_number, _parse_math


NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

KG = r"(?:kg|kilograms?)\b"
METER = r"(?:m|met(?:er|re)s?)(?!\s*/\s*s)"
MPS2 = r"(?:m\s*/\s*s\^2|m\s*/\s*s\s+squared|met(?:er|re)s?\s+per\s+second\s+squared)"
MPS = r"(?:m\s*/\s*s(?!\^?2)|met(?:er|re)s?\s+per\s+second(?!\s+squared))"
SECOND = r"(?:s|sec|secs|seconds?)\b"
NEWTON = r"(?:n|newtons?)\b"
JOULE = r"(?:j|joules?)\b"
WATT = r"(?:w|watts?|j/s)\b"
OHM = r"(?:ohms?|ohm)\b"
VOLT = r"(?:v|volts?)\b"
AMP = r"(?:a|amps?|amperes?)\b"
MOLE = r"(?:mol|moles?)\b"
GRAM = r"(?:g|grams?)\b"
LITER = r"(?:l|lit(?:er|re)s?)\b"
KELVIN = r"(?:k|kelvin)\b"
ATM = r"(?:atm|atmospheres?)\b"
MOLAR = r"(?:m|mol\s*/\s*l|molar)\b"

# Rounded school-level atomic masses keep examples such as 18 g of water
# equal to 1 mol while staying accurate enough for the supported calculations.
ATOMIC_MASS: Dict[str, float] = {
    "H": 1.0, "C": 12.0, "N": 14.0, "O": 16.0, "Na": 23.0,
    "Mg": 24.0, "Al": 27.0, "P": 31.0, "S": 32.0, "Cl": 35.5,
    "K": 39.0, "Ca": 40.0, "Fe": 56.0, "Cu": 64.0, "Zn": 65.0,
    "Br": 80.0, "Ag": 108.0, "I": 127.0, "Ba": 137.0, "Si": 28.0,
}

COMPOUND_ALIASES: Dict[str, str] = {
    "water": "H2O",
    "salt": "NaCl",
    "sodium chloride": "NaCl",
    "carbon dioxide": "CO2",
    "sulphuric acid": "H2SO4",
    "sulfuric acid": "H2SO4",
    "hydrochloric acid": "HCl",
    "sodium hydroxide": "NaOH",
    "calcium hydroxide": "Ca(OH)2",
    "ammonia": "NH3",
    "methane": "CH4",
    "glucose": "C6H12O6",
    "ethanol": "C2H6O",
}

STRONG_ACIDS: Dict[str, int] = {"hcl": 1, "hbr": 1, "hi": 1, "hno3": 1, "h2so4": 1}
STRONG_BASES: Dict[str, int] = {"naoh": 1, "koh": 1, "ca(oh)2": 2, "mg(oh)2": 2}

SUBJECT_CUES: Dict[str, str] = {
    "math": r"\b(?:solve|simplify|factor|expand|equation|roots?|quadratic)\b",
    "physics": (
        r"\b(?:force|mass|accelerat\w*|velocity|speed|distance|displacement|work|"
        r"energy|power|watt|joule|newton|ohm|resistor|voltage|current|pressure|"
        r"volume|temperature|kelvin|ideal|gas|kinematic|momentum|heat)\b|\bm\s*/\s*s"
    ),
    "chemistry": (
        r"\b(?:mole|molar|ph|acid|base|stoich\w*|reaction|water|yield\w*|"
        r"concentration|solution|h2o|co2|nacl|hcl|naoh|h2so4)\b"
    ),
}


@dataclass(frozen=True)
class SolverMatch:
    """A routed solver result, kept separate from card/Wikipedia answers."""

    subject: str
    kind: str
    answer: str


def _normalize(text: str) -> str:
    value = str(text or "").lower()
    value = value.replace("\u2212", "-").replace("\u2013", "-")
    value = value.replace("\u00b2", "^2").replace("\u00b3", "^3")
    value = value.replace("\u00d7", "*").replace("\u00f7", "/")
    value = re.sub(r"\bm\s*/\s*s\^2\b", "m/s^2", value)
    value = re.sub(r"\bm\s*/\s*s\b(?!\^?2)", "m/s", value)
    return re.sub(r"\s+", " ", value).strip(" .?!")


def _fmt(value: float, digits: int = 6) -> str:
    if not math.isfinite(value):
        return "undefined"
    return f"{value:.{digits}g}"


def _unit_value(text: str, unit: str, labels: Sequence[str] = ()) -> Optional[float]:
    """Read a number attached to a unit, optionally near a named quantity."""
    for label in labels:
        match = re.search(rf"{label}[^\d]{{0,35}}?({NUMBER})\s*{unit}", text, re.I)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    match = re.search(rf"({NUMBER})\s*{unit}", text, re.I)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _has_computation(question: str) -> bool:
    return bool(re.search(r"\d|=", question) or re.search(
        r"\b(?:simplify|factor|expand)\b.*\b[xn]\b", question, re.I
    ))


def _detect_subject(question: str, classification: Optional[Classification]) -> Optional[str]:
    if classification and classification.subject_key in ("math", "physics", "chemistry"):
        return classification.subject_key

    scores = {
        subject: len(re.findall(pattern, question, re.I))
        for subject, pattern in SUBJECT_CUES.items()
    }
    if re.search(r"\b(?:solve|simplify|factor|expand)\b|\bx\s*[+*/^]", question, re.I):
        scores["math"] += 2
    if re.search(rf"{NUMBER}\s*(?:{MPS}|{MPS2}|{KG}|{NEWTON}|{JOULE}|{WATT}|{OHM})", question, re.I):
        scores["physics"] += 2
    if re.search(rf"{NUMBER}\s*(?:{MOLE}|{GRAM}|{MOLAR})", question, re.I):
        scores["chemistry"] += 2

    best = max(scores, key=lambda subject: (scores[subject], subject == "physics"))
    return best if scores[best] else None


def _sympy_value(value) -> str:
    if getattr(value, "is_number", False) and getattr(value, "is_real", False):
        return _format_number(float(N(value, 12)))
    return str(value)


def _solve_math(question: str) -> Optional[str]:
    raw = str(question or "").strip()
    if "=" in raw:
        source = re.sub(
            r"^\s*(?:please\s+)?(?:solve\s+for\s+\w+\s*[:=]?\s*|solve|roots\s+of|find\s+x\s+in)\s+",
            "",
            raw,
            flags=re.I,
        ).strip(" .?!")
        left_text, right_text = source.split("=", 1)
        try:
            equation = Eq(_parse_math(left_text), _parse_math(right_text))
            variables = sorted(equation.free_symbols, key=lambda symbol: symbol.name)
            if not variables:
                return f"That equation reduces to {simplify(equation.lhs - equation.rhs)} = 0."
            variable = next((symbol for symbol in variables if symbol.name == "x"), variables[0])
            solutions = sympy_solve(equation, variable)
            if not solutions:
                return f"There is no solution for {variable}."
            rendered = ", ".join(_sympy_value(value) for value in solutions)
            return f"Solving {equation.lhs} = {equation.rhs} gives {variable} = {rendered}."
        except Exception:
            return None

    operation = re.match(r"\s*(?:please\s+)?(simplify|factor|expand)\s+(.+?)\s*$", raw, re.I)
    if operation:
        name, source = operation.group(1).lower(), operation.group(2)
        functions = {"simplify": simplify, "factor": factor, "expand": expand}
        verbs = {"simplify": "Simplified", "factor": "Factored", "expand": "Expanded"}
        try:
            result = functions[name](_parse_math(source))
            return f"{verbs[name]}, that is {result}."
        except Exception:
            return None
    return None


def _kinematics(question: str) -> Optional[str]:
    q = _normalize(question)
    if not re.search(r"\d", q) or not re.search(
        r"m/s|second|distance|displacement|velocity|speed|travels?|accelerat", q, re.I
    ):
        return None

    if re.search(r"from\s+rest", q, re.I):
        initial = 0.0
    else:
        initial = _unit_value(q, MPS, [r"initial\s+(?:velocity|speed)", r"\bu\b"])
    speed = _unit_value(q, MPS, [r"travels?", r"speed", r"velocity", r"moving", r"\bv\b"])
    if initial is None:
        initial = speed
    final = _unit_value(q, MPS, [r"final\s+velocity", r"reaches?", r"ends?\s+(?:up\s+)?at"])
    acceleration = _unit_value(q, MPS2, [r"accelerat\w*", r"acceleration", r"\ba\b"])
    time = _unit_value(q, SECOND, [r"for", r"time", r"after", r"during", r"\bt\b"])
    distance = _unit_value(q, METER, [r"distance", r"displacement", r"how far", r"travels?"])

    asks_final = bool(re.search(r"final\s+velocity|velocity\s+after|speed\s+after|how\s+fast", q, re.I))
    asks_distance = bool(re.search(r"distance|displacement|how\s+far|how\s+much.*travel", q, re.I))
    asks_acceleration = bool(re.search(r"acceleration|accelerat\w*", q, re.I))
    asks_time = bool(re.search(r"how\s+long|time\s+taken", q, re.I))

    if asks_final and final is None and initial is not None and acceleration is not None and time is not None:
        value = initial + acceleration * time
        return (
            f"Using v = u + at, the final velocity is "
            f"{_fmt(initial)} + {_fmt(acceleration)} * {_fmt(time)} = {_fmt(value)} m/s."
        )

    if asks_distance and distance is None and time is not None:
        if initial is not None and acceleration is not None:
            value = initial * time + 0.5 * acceleration * time * time
            return (
                f"Using s = ut + 1/2at^2, the distance is {_fmt(value)} m "
                f"(u = {_fmt(initial)} m/s, a = {_fmt(acceleration)} m/s^2, t = {_fmt(time)} s)."
            )
        if speed is not None:
            value = speed * time
            return (
                f"Using s = v * t, the distance is {_fmt(speed)} * {_fmt(time)} = {_fmt(value)} m."
            )

    if asks_acceleration and acceleration is None and final is not None and initial is not None and time:
        value = (final - initial) / time
        return (
            f"Using a = (v - u) / t, the acceleration is "
            f"({_fmt(final)} - {_fmt(initial)}) / {_fmt(time)} = {_fmt(value)} m/s^2."
        )

    if asks_time and time is None and distance is not None and speed:
        value = distance / speed
        return f"Using t = s / v, the time is {_fmt(distance)} / {_fmt(speed)} = {_fmt(value)} s."
    return None


def _newtons_second_law(question: str) -> Optional[str]:
    q = _normalize(question)
    if not re.search(r"\bforce\b|\bmass\b|accelerat|newton", q, re.I):
        return None
    mass = _unit_value(q, KG, [r"mass", r"\bm\b"])
    force = _unit_value(q, NEWTON, [r"force", r"\bf\b"])
    acceleration = _unit_value(q, MPS2, [r"accelerat\w*", r"acceleration", r"\ba\b"])

    asks_force = bool(re.search(r"\bforce\b|newton", q, re.I))
    asks_acceleration = bool(re.search(r"acceleration|accelerat\w*", q, re.I))
    asks_mass = bool(re.search(r"\bmass\b|kilogram", q, re.I)) and not re.search(
        r"\d+\s*kg", q, re.I
    )

    if mass is not None and acceleration is not None and (force is None or asks_force):
        value = mass * acceleration
        return (
            f"Using F = ma, the force is {_fmt(mass)} * {_fmt(acceleration)} = {_fmt(value)} N."
        )
    if force is not None and mass is not None and asks_acceleration and acceleration is None:
        value = force / mass
        return f"Using a = F / m, the acceleration is {_fmt(force)} / {_fmt(mass)} = {_fmt(value)} m/s^2."
    if force is not None and acceleration is not None and asks_mass and mass is None:
        value = force / acceleration
        return f"Using m = F / a, the mass is {_fmt(force)} / {_fmt(acceleration)} = {_fmt(value)} kg."
    return None


def _work_energy_power(question: str) -> Optional[str]:
    q = _normalize(question)
    if not re.search(r"\bwork\b|energy|power|watt|joule|kinetic|potential", q, re.I):
        return None
    force = _unit_value(q, NEWTON, [r"force", r"\bf\b"])
    distance = _unit_value(q, METER, [r"distance", r"over", r"through", r"displacement"])
    work = _unit_value(q, JOULE, [r"work", r"energy", r"\bw\b"])
    time = _unit_value(q, SECOND, [r"time", r"in", r"over", r"\bt\b"])
    mass = _unit_value(q, KG, [r"mass", r"\bm\b"])
    velocity = _unit_value(q, MPS, [r"velocity", r"speed", r"\bv\b"])
    height = _unit_value(q, METER, [r"height", r"above", r"\bh\b"])

    asks_power = bool(re.search(r"\bpower\b|watt", q, re.I))
    asks_kinetic = bool(re.search(r"kinetic|\bke\b", q, re.I))
    asks_potential = bool(re.search(r"potential|\bgpe\b", q, re.I))
    asks_work = bool(re.search(r"\bwork\b|joule", q, re.I))

    if asks_power and time:
        if work is not None:
            value = work / time
            return f"Using P = W / t, the power is {_fmt(work)} / {_fmt(time)} = {_fmt(value)} W."
        if force is not None and distance is not None:
            work_value = force * distance
            value = work_value / time
            return (
                f"The work is F * d = {_fmt(force)} * {_fmt(distance)} = {_fmt(work_value)} J, "
                f"so P = W / t = {_fmt(value)} W."
            )

    if asks_kinetic and mass is not None and velocity is not None:
        value = 0.5 * mass * velocity * velocity
        return (
            f"Using KE = 1/2mv^2, the kinetic energy is 1/2 * {_fmt(mass)} * "
            f"{_fmt(velocity)}^2 = {_fmt(value)} J."
        )

    if asks_potential and mass is not None and height is not None:
        value = mass * 9.8 * height
        return (
            f"Using PE = mgh with g = 9.8 m/s^2, the potential energy is "
            f"{_fmt(mass)} * 9.8 * {_fmt(height)} = {_fmt(value)} J."
        )

    if asks_work and work is None and force is not None and distance is not None:
        value = force * distance
        return f"Using W = F * d, the work done is {_fmt(force)} * {_fmt(distance)} = {_fmt(value)} J."
    return None


def _ohms_law(question: str) -> Optional[str]:
    q = _normalize(question)
    if not re.search(r"ohm|resistor|current|voltage|potential difference", q, re.I):
        return None
    voltage = _unit_value(q, VOLT, [r"voltage", r"potential difference", r"\bv\b"])
    resistance = _unit_value(q, OHM, [r"resistance", r"resistor", r"\br\b"])
    current = _unit_value(q, AMP, [r"current", r"\bi\b"])

    asks_current = bool(re.search(r"current|amp", q, re.I))
    asks_resistance = bool(re.search(r"resistance|resistor|ohm", q, re.I))
    asks_voltage = bool(re.search(r"voltage|potential difference|volt", q, re.I))

    if asks_current and current is None and voltage is not None and resistance is not None:
        value = voltage / resistance
        return f"Using I = V / R, the current is {_fmt(voltage)} / {_fmt(resistance)} = {_fmt(value)} A."
    if asks_resistance and resistance is None and voltage is not None and current is not None:
        value = voltage / current
        return f"Using R = V / I, the resistance is {_fmt(voltage)} / {_fmt(current)} = {_fmt(value)} ohm."
    if asks_voltage and voltage is None and current is not None and resistance is not None:
        value = current * resistance
        return f"Using V = IR, the voltage is {_fmt(current)} * {_fmt(resistance)} = {_fmt(value)} V."
    return None


def _ideal_gas_law(question: str) -> Optional[str]:
    q = _normalize(question)
    moles = _unit_value(q, MOLE, [r"\bn\b", r"moles?"])
    temperature = _unit_value(q, KELVIN, [r"\bt\b", r"temperature"])
    volume = _unit_value(q, LITER, [r"\bv\b", r"volume"])
    pressure = _unit_value(q, ATM, [r"\bp\b", r"pressure"])
    known = sum(value is not None for value in (moles, temperature, volume, pressure))
    if known < 3 and not re.search(r"ideal gas|pv\s*=?\s*nrt", q, re.I):
        return None

    gas_constant = 0.082057  # L atm/(mol K)
    unit_note = f"using R = {gas_constant} L atm/(mol K)"

    if re.search(r"pressure|\bp\b", q, re.I) and pressure is None:
        if None in (moles, temperature, volume) or not volume:
            return None
        value = moles * gas_constant * temperature / volume
        return f"Using PV = nRT, the pressure is {_fmt(value, 4)} atm ({unit_note})."
    if re.search(r"volume|\bv\b", q, re.I) and volume is None:
        if None in (moles, temperature, pressure) or not pressure:
            return None
        value = moles * gas_constant * temperature / pressure
        return f"Using PV = nRT, the volume is {_fmt(value, 4)} L ({unit_note})."
    if re.search(r"temperature|\bt\b", q, re.I) and temperature is None:
        if None in (moles, pressure, volume) or not moles:
            return None
        value = pressure * volume / (moles * gas_constant)
        return f"Using PV = nRT, the temperature is {_fmt(value, 4)} K ({unit_note})."
    if re.search(r"moles?|\bn\b", q, re.I) and moles is None:
        if None in (pressure, volume, temperature) or not temperature:
            return None
        value = pressure * volume / (gas_constant * temperature)
        return f"Using PV = nRT, the amount is {_fmt(value, 4)} mol ({unit_note})."
    return None


def _solve_physics(question: str) -> Optional[str]:
    return (
        _kinematics(question)
        or _newtons_second_law(question)
        or _work_energy_power(question)
        or _ohms_law(question)
        or _ideal_gas_law(question)
    )


def _molar_mass(formula: str) -> float:
    """Compute molar mass from a formula, including simple parentheses."""
    total = 0.0
    multiplier = 1
    stack: List[Tuple[float, int]] = []
    index = 0
    text = str(formula or "").strip()

    while index < len(text):
        char = text[index]
        if char == "(":
            stack.append((total, multiplier))
            total, multiplier = 0.0, 1
            index += 1
            continue
        if char == ")":
            index += 1
            digits = ""
            while index < len(text) and text[index].isdigit():
                digits += text[index]
                index += 1
            factor = int(digits) if digits else 1
            previous_total, previous_multiplier = stack.pop()
            total = previous_total + previous_multiplier * factor * total
            multiplier = previous_multiplier
            continue
        if char.isupper():
            symbol = char
            index += 1
            while index < len(text) and text[index].islower():
                symbol += text[index]
                index += 1
            digits = ""
            while index < len(text) and text[index].isdigit():
                digits += text[index]
                index += 1
            if symbol not in ATOMIC_MASS:
                raise ValueError(f"Unknown element: {symbol}")
            total += multiplier * ATOMIC_MASS[symbol] * (int(digits) if digits else 1)
            continue
        if char.isdigit():
            raise ValueError("Unexpected digit in formula")
        index += 1

    if total <= 0 or stack:
        raise ValueError("Invalid formula")
    return total


def _canonical_formula(text: str) -> Optional[str]:
    value = str(text or "").strip()
    low = _normalize(value)
    for name, formula in COMPOUND_ALIASES.items():
        if re.search(rf"\b{re.escape(name)}\b", low, re.I):
            return formula
    for token in re.findall(r"[A-Za-z][A-Za-z0-9()]*", value):
        try:
            _molar_mass(token)
        except (ValueError, IndexError):
            continue
        return token
    return None


def _ph(question: str) -> Optional[str]:
    raw = str(question or "")
    low = _normalize(raw)
    if not re.search(r"\bph\b|acid|base|hcl|hbr|hi|hno3|h2so4|naoh|koh|ca\(oh\)2", low):
        return None
    match = re.search(rf"({NUMBER})\s*(?:m|mol\s*/\s*l|molar)\b", low, re.I)
    if not match:
        return None
    concentration = float(match.group(1))
    if concentration <= 0:
        return None

    for name, protons in STRONG_ACIDS.items():
        if re.search(rf"\b{re.escape(name)}\b", low, re.I):
            value = -math.log10(concentration * protons)
            return (
                f"For the strong acid {name.upper()}, pH = -log[H+] = "
                f"-log({_fmt(concentration, 4)}) = {_fmt(value, 4)}."
            )
    for name, hydroxides in STRONG_BASES.items():
        if re.search(rf"\b{re.escape(name)}\b", low, re.I):
            hydroxide = concentration * hydroxides
            value = 14.0 + math.log10(hydroxide)
            return (
                f"For the strong base {name.upper()}, [OH-] = {_fmt(hydroxide, 4)} M, "
                f"so pH = {_fmt(value, 4)}."
            )
    return None


def _moles_and_molar_mass(question: str) -> Optional[str]:
    q = _normalize(question)
    formula = _canonical_formula(question)
    if not formula:
        return None
    try:
        molar_mass = _molar_mass(formula)
    except (ValueError, IndexError):
        return None

    if re.search(r"molar\s+mass|g\s*/\s*mol", q, re.I):
        return f"The molar mass of {formula} is {_fmt(molar_mass, 4)} g/mol."

    grams = _unit_value(q, GRAM, [r"mass", r"grams?"])
    moles = _unit_value(q, MOLE, [r"moles?", r"\bn\b"])
    if grams is not None and moles is None:
        value = grams / molar_mass
        return (
            f"{_fmt(grams, 4)} g of {formula} is {_fmt(value, 4)} mol "
            f"(molar mass {_fmt(molar_mass, 4)} g/mol)."
        )
    if moles is not None and grams is None:
        value = moles * molar_mass
        return (
            f"{_fmt(moles, 4)} mol of {formula} has a mass of {_fmt(value, 4)} g "
            f"(molar mass {_fmt(molar_mass, 4)} g/mol)."
        )
    return None


def _parse_reaction_side(side: str) -> List[Tuple[float, str]]:
    entries: List[Tuple[float, str]] = []
    for term in side.split("+"):
        match = re.match(rf"^\s*({NUMBER})?\s*([A-Z][A-Za-z0-9()]*)\s*$", term.strip())
        if not match:
            continue
        coefficient = float(match.group(1)) if match.group(1) else 1.0
        entries.append((coefficient, match.group(2)))
    return entries


def _reaction_display_name(formula: str) -> str:
    for name, known in COMPOUND_ALIASES.items():
        if known.lower() == formula.lower():
            return name
    return formula


def _stoichiometry(question: str) -> Optional[str]:
    raw = re.sub(r"\byields?\b", "->", str(question or ""), flags=re.I)
    arrow = re.search(r"->|=>", raw)
    if not arrow:
        return None

    # Only read the formula expression around the arrow; the surrounding
    # question text would otherwise be mistaken for the reactant side.
    term = r"(?:\d+\s*)?[A-Z][A-Za-z0-9()]*"
    side = rf"{term}(?:\s*\+\s*{term})*"
    left_match = re.search(rf"({side})\s*$", raw[:arrow.start()])
    right_match = re.match(rf"\s*({side})", raw[arrow.end():])
    if not left_match or not right_match:
        return None
    reactants = _parse_reaction_side(left_match.group(1))
    products = _parse_reaction_side(right_match.group(1))
    if not reactants or not products:
        return None

    amount_match = re.search(
        rf"({NUMBER})\s*(?:moles?|mol)\s+(?:of\s+)?([A-Za-z][A-Za-z0-9()]*|water|salt|ammonia)",
        raw,
        re.I,
    )
    if not amount_match:
        return None
    amount = float(amount_match.group(1))
    source = _canonical_formula(amount_match.group(2)) or amount_match.group(2)

    target = None
    for _, formula in products:
        if re.search(rf"\b{re.escape(formula)}\b", raw, re.I) or re.search(
            rf"\b{re.escape(_reaction_display_name(formula))}\b", raw, re.I
        ):
            target = formula
            if target.lower() != source.lower():
                break
    if target is None:
        target = products[0][1]

    source_entry = next((item for item in reactants if item[1].lower() == source.lower()), None)
    target_entry = next((item for item in products if item[1].lower() == target.lower()), None)
    if not source_entry or not target_entry or source_entry[0] <= 0:
        return None

    value = amount * target_entry[0] / source_entry[0]
    source_name = _reaction_display_name(source_entry[1])
    target_name = _reaction_display_name(target_entry[1])
    return (
        f"The balanced equation gives a {source_entry[1]}-to-{target_entry[1]} ratio of "
        f"{_fmt(source_entry[0], 4)}:{_fmt(target_entry[0], 4)}, so {_fmt(amount, 4)} mol "
        f"{source_name} produces {_fmt(value, 4)} mol {target_name} ({target_entry[1]})."
    )


def _solve_chemistry(question: str) -> Optional[str]:
    return (
        _stoichiometry(question)
        or _ph(question)
        or _moles_and_molar_mass(question)
        or _ideal_gas_law(question)
    )


def solve(question: str, classification: Optional[Classification] = None) -> Optional[str]:
    """Return an applied Math/Physics/Chemistry answer when the question computes."""
    raw = str(question or "").strip()
    if not raw or not _has_computation(raw):
        return None
    subject = _detect_subject(_normalize(raw), classification)
    if subject == "math":
        return _solve_math(raw)
    if subject == "physics":
        return _solve_physics(raw)
    if subject == "chemistry":
        return _solve_chemistry(raw)
    return None


__all__ = [
    "SolverMatch",
    "solve",
]
