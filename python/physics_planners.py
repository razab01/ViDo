"""Deterministic physics intent planners.

Retrieval understands the question; this module only recognises a narrow
numerical intent and lets plain Python compute the result. It has no dependency
on the other planner modules, so it can run before any answer routing. Replies
are badge-free and work fully offline.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
Params = Dict[str, Any]
Compute = Callable[[Params], str]

G = 6.674e-11
G0 = 9.8
C_LIGHT = 3.0e8
H_PLANCK = 6.626e-34
HBAR = 1.055e-34
K_B = 1.381e-23
R_GAS = 8.314
K_E = 8.988e9
M_ELECTRON = 9.109e-31
EPS0 = 8.854e-12
E_CHARGE = 1.602e-19
FARADAY = 96485.0


@dataclass(frozen=True)
class PhysicsPlan:
    intent: str
    params: Params
    compute: Compute


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
    if value == 0 or value == -0.0:
        return "0"
    text = f"{value:.{digits}g}"
    return re.sub(r"e([+-])0", r"e\1", text)


def _num(q: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, q, re.I)
    if not match:
        return None
    raw = next((group for group in match.groups() if group is not None), None)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _nums(q: str, pattern: str) -> List[float]:
    values: List[float] = []
    for match in re.finditer(pattern, q, re.I):
        try:
            values.append(float(match.group(1)))
        except (TypeError, ValueError):
            continue
    return values


def _all(q: str, *patterns: str) -> bool:
    return all(re.search(pattern, q, re.I) for pattern in patterns)


def _formula(formula: str, substitution: str, value: float, unit: str) -> str:
    return f"Using {formula}, {substitution} = {_fmt(value)} {unit}."


def _compute_friction(p: Params) -> str:
    value = p["mu"] * p["mass"] * G0
    return _formula(
        "F = μmg",
        f"{_fmt(p['mu'])} × {_fmt(p['mass'])} × {_fmt(G0)}",
        value,
        "N",
    )


def _compute_momentum(p: Params) -> str:
    value = p["mass"] * p["speed"]
    return _formula("p = mv", f"{_fmt(p['mass'])} × {_fmt(p['speed'])}", value, "kg m/s")


def _compute_impulse(p: Params) -> str:
    value = p["force"] * p["time"]
    return _formula("J = Ft", f"{_fmt(p['force'])} × {_fmt(p['time'])}", value, "N s")


def _compute_elastic_collision(p: Params) -> str:
    return (
        f"For an elastic collision between equal masses with one cart initially at rest, "
        f"the moving cart stops and the second cart leaves at {_fmt(p['speed'])} m/s."
    )


def _compute_inelastic_collision(p: Params) -> str:
    value = p["m1"] * p["u1"] / (p["m1"] + p["m2"])
    return _formula(
        "v = m₁u₁/(m₁ + m₂)",
        f"{_fmt(p['m1'])} × {_fmt(p['u1'])}/({_fmt(p['m1'])} + {_fmt(p['m2'])})",
        value,
        "m/s",
    )


def _compute_centripetal(p: Params) -> str:
    value = p["mass"] * p["speed"] ** 2 / p["radius"]
    return _formula(
        "F = mv²/r",
        f"{_fmt(p['mass'])} × {_fmt(p['speed'])}²/{_fmt(p['radius'])}",
        value,
        "N",
    )


def _compute_orbital_speed(p: Params) -> str:
    value = math.sqrt(G * p["mass"] / p["radius"])
    return _formula(
        "v = √(GM/r)",
        f"√({_fmt(G)} × {_fmt(p['mass'])}/{_fmt(p['radius'])})",
        value,
        "m/s",
    )


def _compute_gravity_force(p: Params) -> str:
    value = G * p["m1"] * p["m2"] / p["radius"] ** 2
    return _formula(
        "F = Gm₁m₂/r²",
        f"{_fmt(G)} × {_fmt(p['m1'])} × {_fmt(p['m2'])}/{_fmt(p['radius'])}²",
        value,
        "N",
    )


def _compute_gravity_energy(p: Params) -> str:
    value = G * p["m1"] * p["m2"] / p["radius"]
    return _formula(
        "U = Gm₁m₂/r",
        f"{_fmt(G)} × {_fmt(p['m1'])} × {_fmt(p['m2'])}/{_fmt(p['radius'])}",
        value,
        "J",
    )


def _compute_projectile_range(p: Params) -> str:
    angle = math.radians(p["angle"])
    value = p["speed"] ** 2 * math.sin(2 * angle) / G0
    return _formula(
        "R = v²sin(2θ)/g",
        f"{_fmt(p['speed'])}² × sin(2 × {_fmt(p['angle'])}°)/{_fmt(G0)}",
        value,
        "m",
    )


def _compute_projectile_height(p: Params) -> str:
    angle = math.radians(p["angle"])
    value = (p["speed"] * math.sin(angle)) ** 2 / (2 * G0)
    return _formula(
        "H = v²sin²θ/(2g)",
        f"({_fmt(p['speed'])} × sin({_fmt(p['angle'])}°))²/(2 × {_fmt(G0)})",
        value,
        "m",
    )


def _compute_projectile_time(p: Params) -> str:
    angle = math.radians(p["angle"])
    value = 2 * p["speed"] * math.sin(angle) / G0
    return _formula(
        "T = 2v sinθ/g",
        f"2 × {_fmt(p['speed'])} × sin({_fmt(p['angle'])}°)/{_fmt(G0)}",
        value,
        "s",
    )


def _compute_spring_period(p: Params) -> str:
    value = 2 * math.pi * math.sqrt(p["mass"] / p["spring_constant"])
    return _formula(
        "T = 2π√(m/k)",
        f"2π√({_fmt(p['mass'])}/{_fmt(p['spring_constant'])})",
        value,
        "s",
    )


def _compute_pendulum_period(p: Params) -> str:
    value = 2 * math.pi * math.sqrt(p["length"] / G0)
    return _formula(
        "T = 2π√(L/g)",
        f"2π√({_fmt(p['length'])}/{_fmt(G0)})",
        value,
        "s",
    )


def _compute_wave_speed(p: Params) -> str:
    value = p["frequency"] * p["wavelength"]
    return _formula(
        "v = fλ",
        f"{_fmt(p['frequency'])} × {_fmt(p['wavelength'])}",
        value,
        "m/s",
    )


def _compute_doppler_observer(p: Params) -> str:
    value = p["frequency"] * (p["sound_speed"] + p["observer_speed"]) / p["sound_speed"]
    return _formula(
        "f' = f(v + v₀)/v",
        f"{_fmt(p['frequency'])} × ({_fmt(p['sound_speed'])} + {_fmt(p['observer_speed'])})/{_fmt(p['sound_speed'])}",
        value,
        "Hz",
    )


def _compute_doppler_source(p: Params) -> str:
    denominator = p["sound_speed"] - p["source_speed"]
    if denominator <= 0:
        return "The source speed reaches the wave speed, so this simple Doppler formula has no finite frequency ahead of the source."
    value = p["frequency"] * p["sound_speed"] / denominator
    return _formula(
        "f' = fv/(v - vₛ)",
        f"{_fmt(p['frequency'])} × {_fmt(p['sound_speed'])}/({_fmt(p['sound_speed'])} - {_fmt(p['source_speed'])})",
        value,
        "Hz",
    )


def _compute_snell(p: Params) -> str:
    ratio = math.sin(math.radians(p["incident"])) / p["index"]
    if abs(ratio) > 1:
        return (
            f"Snell's law gives sin(r) = sin({_fmt(p['incident'])}°)/{_fmt(p['index'])} > 1, "
            "so the ray undergoes total internal reflection and there is no refracted angle."
        )
    value = math.degrees(math.asin(ratio))
    return _formula(
        "n₁sinθ₁ = n₂sinθ₂",
        f"arcsin(sin({_fmt(p['incident'])}°)/{_fmt(p['index'])})",
        value,
        "°",
    )


def _compute_lens(p: Params) -> str:
    denominator = 1 / p["focal"] - 1 / p["object_distance"]
    if abs(denominator) < 1e-15:
        return "The thin-lens equation puts the image at infinity because the object is at the focal point."
    value = 1 / denominator
    return _formula(
        "1/f = 1/u + 1/v",
        f"1/({_fmt(1 / p['focal'])} - {_fmt(1 / p['object_distance'])})",
        value,
        "m",
    )


def _compute_coulomb(p: Params) -> str:
    value = K_E * p["q1"] * p["q2"] / p["radius"] ** 2
    return _formula(
        "F = kq₁q₂/r²",
        f"{_fmt(K_E)} × {_fmt(p['q1'])} × {_fmt(p['q2'])}/{_fmt(p['radius'])}²",
        value,
        "N",
    )


def _compute_electric_field(p: Params) -> str:
    value = K_E * p["charge"] / p["radius"] ** 2
    return _formula(
        "E = kq/r²",
        f"{_fmt(K_E)} × {_fmt(p['charge'])}/{_fmt(p['radius'])}²",
        value,
        "N/C",
    )


def _compute_electric_potential(p: Params) -> str:
    value = K_E * p["charge"] / p["radius"]
    return _formula(
        "V = kq/r",
        f"{_fmt(K_E)} × {_fmt(p['charge'])}/{_fmt(p['radius'])}",
        value,
        "V",
    )


def _compute_parallel_resistors(p: Params) -> str:
    value = p["r1"] * p["r2"] / (p["r1"] + p["r2"])
    return _formula(
        "R_eq = R₁R₂/(R₁ + R₂)",
        f"{_fmt(p['r1'])} × {_fmt(p['r2'])}/({_fmt(p['r1'])} + {_fmt(p['r2'])})",
        value,
        "Ω",
    )


def _compute_series_resistors(p: Params) -> str:
    value = p["r1"] + p["r2"]
    return _formula("R_eq = R₁ + R₂", f"{_fmt(p['r1'])} + {_fmt(p['r2'])}", value, "Ω")


def _compute_capacitor_energy(p: Params) -> str:
    value = 0.5 * p["capacitance"] * p["voltage"] ** 2
    return _formula(
        "E = 1/2CV²",
        f"1/2 × {_fmt(p['capacitance'])} × {_fmt(p['voltage'])}²",
        value,
        "J",
    )


def _compute_emf(p: Params) -> str:
    value = p["turns"] * p["area"] * abs(p["b2"] - p["b1"]) / p["time"]
    return _formula(
        "emf = N A |ΔB|/Δt",
        f"{p['turns']} × {_fmt(p['area'])} × |{_fmt(p['b2'])} - {_fmt(p['b1'])}|/{_fmt(p['time'])}",
        value,
        "V",
    )


def _compute_rms(p: Params) -> str:
    value = p["peak"] / math.sqrt(2)
    return _formula(
        f"{p['kind']}_rms = {p['kind']}_peak/√2",
        f"{_fmt(p['peak'])}/√2",
        value,
        p["unit"],
    )


def _compute_heat(p: Params) -> str:
    value = p["mass"] * p["specific_heat"] * p["delta_t"]
    return _formula(
        "Q = mcΔT",
        f"{_fmt(p['mass'])} × {_fmt(p['specific_heat'])} × {_fmt(p['delta_t'])}",
        value,
        "J",
    )


def _compute_entropy(p: Params) -> str:
    value = p["moles"] * R_GAS * math.log(p["v2"] / p["v1"])
    return _formula(
        "ΔS = nR ln(V₂/V₁)",
        f"{_fmt(p['moles'])} × {_fmt(R_GAS)} × ln({_fmt(p['v2'])}/{_fmt(p['v1'])})",
        value,
        "J/K",
    )


def _compute_photoelectric(p: Params) -> str:
    photon_energy = H_PLANCK * p["frequency"]
    value = max(0.0, photon_energy - p["work_function"])
    return (
        f"Photon energy is hf = {_fmt(photon_energy)} J, so the maximum photoelectron kinetic "
        f"energy is {_fmt(value)} J."
    )


def _compute_de_broglie(p: Params) -> str:
    value = H_PLANCK / p["momentum"]
    return _formula(
        "λ = h/p",
        f"{_fmt(H_PLANCK)}/{_fmt(p['momentum'])}",
        value,
        "m",
    )


def _compute_uncertainty(p: Params) -> str:
    value = HBAR / (2 * p["position"])
    return _formula(
        "Δp ≥ ℏ/(2Δx)",
        f"{_fmt(HBAR)}/(2 × {_fmt(p['position'])})",
        value,
        "kg m/s",
    )


def _compute_rest_energy(p: Params) -> str:
    value = p["mass"] * C_LIGHT ** 2
    return _formula(
        "E = mc²",
        f"{_fmt(p['mass'])} × {_fmt(C_LIGHT)}²",
        value,
        "J",
    )


def _compute_time_dilation(p: Params) -> str:
    ratio = p["speed"] / C_LIGHT
    if ratio >= 1:
        return "A massive object cannot reach or exceed the speed of light, so this time dilation has no physical value."
    gamma = 1 / math.sqrt(1 - ratio ** 2)
    value = gamma * p["proper_time"]
    return (
        f"γ = 1/√(1 - v²/c²) = {_fmt(gamma)}. The measured time is γt₀ = "
        f"{_fmt(gamma)} × {_fmt(p['proper_time'])} = {_fmt(value)} s."
    )


def _compute_decay(p: Params) -> str:
    value = p["initial"] * 2 ** (-p["elapsed"] / p["half_life"])
    return _formula(
        "N = N₀ × 2^(-t/T)",
        f"{_fmt(p['initial'])} × 2^(-{_fmt(p['elapsed'])}/{_fmt(p['half_life'])})",
        value,
        "atoms",
    )


def _compute_hubble(p: Params) -> str:
    value = p["hubble"] * p["distance"]
    return _formula(
        "v = H₀d",
        f"{_fmt(p['hubble'])} × {_fmt(p['distance'])}",
        value,
        "km/s",
    )


def _compute_hydrostatic(p: Params) -> str:
    value = p["density"] * G0 * p["depth"]
    return _formula(
        "P = ρgh",
        f"{_fmt(p['density'])} × {_fmt(G0)} × {_fmt(p['depth'])}",
        value,
        "Pa",
    )


def _compute_uncertainty_percent(p: Params) -> str:
    value = 100 * p["absolute"] / p["measured"]
    return _formula(
        "percentage uncertainty = 100 × absolute uncertainty/measured value",
        f"100 × {_fmt(p['absolute'])}/{_fmt(p['measured'])}",
        value,
        "%",
    )


def _plan_friction(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bfriction\b", rf"{NUM}\s*kg", r"\bcoefficient\b"):
        return None
    mass = _num(q, rf"({NUM})\s*kg")
    mu = _num(
        q,
        rf"(?:coefficient\s*(?:of\s*)?(?:kinetic\s*)?friction|friction\s*coefficient)"
        rf"[^\d]{{0,20}}?({NUM})",
    )
    if None in (mass, mu) or not re.search(r"\bforce\b|\bfriction\b", q, re.I):
        return None
    return PhysicsPlan("physics_friction", {"mass": mass, "mu": mu}, _compute_friction)


def _plan_momentum(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bmomentum\b", rf"{NUM}\s*kg", rf"{NUM}\s*m/s"):
        return None
    if re.search(r"\bcollid|\bsticks\b|\bimpulse\b", q, re.I):
        return None
    mass = _num(q, rf"({NUM})\s*kg")
    speed = _num(q, rf"({NUM})\s*m/s")
    if None in (mass, speed):
        return None
    return PhysicsPlan("physics_momentum", {"mass": mass, "speed": speed}, _compute_momentum)


def _plan_impulse(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bimpulse\b", rf"{NUM}\s*n\b", rf"{NUM}\s*s(?:econds?)?\b"):
        return None
    force = _num(q, rf"({NUM})\s*n\b")
    time = _num(q, rf"({NUM})\s*s(?:econds?)?\b")
    if None in (force, time):
        return None
    return PhysicsPlan("physics_impulse", {"force": force, "time": time}, _compute_impulse)


def _plan_elastic_collision(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\belastic\b", r"\bcollis|\bcollid|\bhit\b", r"\bidentical\b|\bsame mass\b", rf"{NUM}\s*m/s"):
        return None
    speed = _num(q, rf"({NUM})\s*m/s")
    if speed is None or not re.search(r"\bsecond\b.*\bfinal\b|\bfinal\b.*\bsecond\b", q, re.I):
        return None
    return PhysicsPlan("physics_elastic", {"speed": speed}, _compute_elastic_collision)


def _plan_inelastic_collision(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bsticks?\s+(?:together|to)\b|\bperfectly\s+inelastic\b", rf"{NUM}\s*kg", rf"{NUM}\s*m/s"):
        return None
    masses = _nums(q, rf"({NUM})\s*kg")
    speed = _num(q, rf"({NUM})\s*m/s")
    if len(masses) < 2 or speed is None:
        return None
    return PhysicsPlan(
        "physics_inelastic",
        {"m1": masses[0], "m2": masses[1], "u1": speed},
        _compute_inelastic_collision,
    )


def _plan_centripetal(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bcentripetal\b", rf"{NUM}\s*kg", rf"{NUM}\s*m/s", rf"{NUM}\s*m\b"):
        return None
    mass = _num(q, rf"({NUM})\s*kg")
    speed = _num(q, rf"({NUM})\s*m/s")
    radius = _num(q, rf"(?:radius\s*(?:is|=|:)?\s*|at a radius of\s*)({NUM})\s*m\b")
    if None in (mass, speed, radius):
        return None
    return PhysicsPlan(
        "physics_centripetal",
        {"mass": mass, "speed": speed, "radius": radius},
        _compute_centripetal,
    )


def _plan_orbital_speed(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\borbital speed\b", rf"{NUM}\s*kg", rf"{NUM}\s*m\b"):
        return None
    mass = _num(q, rf"({NUM})\s*kg")
    radius = _num(q, rf"(?:radius|orbit)\s*(?:is|=|:)?\s*({NUM})\s*m\b")
    if None in (mass, radius) or radius <= 0:
        return None
    return PhysicsPlan("physics_orbital", {"mass": mass, "radius": radius}, _compute_orbital_speed)


def _plan_gravity_force(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bgravitational force\b", rf"{NUM}\s*kg", rf"{NUM}\s*m\b"):
        return None
    masses = _nums(q, rf"({NUM})\s*kg")
    radius = _num(q, rf"(?:separated by|distance of|distance)\s+({NUM})\s*m\b")
    if len(masses) < 2 or radius is None or radius <= 0:
        return None
    return PhysicsPlan(
        "physics_gravity_force",
        {"m1": masses[0], "m2": masses[1], "radius": radius},
        _compute_gravity_force,
    )


def _plan_gravity_energy(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bgravitational potential energy\b", rf"{NUM}\s*kg", rf"{NUM}\s*m\b"):
        return None
    masses = _nums(q, rf"({NUM})\s*kg")
    radius = _num(q, rf"(?:separated by|distance of|distance)\s+({NUM})\s*m\b")
    if len(masses) < 2 or radius is None or radius <= 0:
        return None
    return PhysicsPlan(
        "physics_gravity_energy",
        {"m1": masses[0], "m2": masses[1], "radius": radius},
        _compute_gravity_energy,
    )


def _projectile_values(q: str) -> Optional[Tuple[float, float]]:
    speed = _num(q, rf"({NUM})\s*m/s")
    angle = _num(q, rf"({NUM})\s*(?:degrees|°|\bdeg\b)")
    if None in (speed, angle):
        return None
    return speed, angle


def _plan_projectile_range(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bprojectile\b|\blaunch", r"\brange\b", rf"{NUM}\s*m/s", rf"{NUM}\s*(?:degrees|°)"):
        return None
    values = _projectile_values(q)
    if values is None:
        return None
    speed, angle = values
    return PhysicsPlan(
        "physics_projectile_range",
        {"speed": speed, "angle": angle},
        _compute_projectile_range,
    )


def _plan_projectile_height(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bprojectile\b|\blaunch", r"\bmaximum height\b", rf"{NUM}\s*m/s", rf"{NUM}\s*(?:degrees|°)"):
        return None
    values = _projectile_values(q)
    if values is None:
        return None
    speed, angle = values
    return PhysicsPlan(
        "physics_projectile_height",
        {"speed": speed, "angle": angle},
        _compute_projectile_height,
    )


def _plan_projectile_time(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bprojectile\b|\blaunch", r"\btime of flight\b", rf"{NUM}\s*m/s", rf"{NUM}\s*(?:degrees|°)"):
        return None
    values = _projectile_values(q)
    if values is None:
        return None
    speed, angle = values
    return PhysicsPlan(
        "physics_projectile_time",
        {"speed": speed, "angle": angle},
        _compute_projectile_time,
    )


def _plan_spring_period(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bspring\b", r"\bperiod\b", rf"{NUM}\s*kg", rf"{NUM}\s*n/m\b"):
        return None
    mass = _num(q, rf"({NUM})\s*kg")
    spring_constant = _num(q, rf"({NUM})\s*n/m\b")
    if None in (mass, spring_constant) or spring_constant <= 0:
        return None
    return PhysicsPlan(
        "physics_spring_period",
        {"mass": mass, "spring_constant": spring_constant},
        _compute_spring_period,
    )


def _plan_pendulum_period(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bpendulum\b", r"\bperiod\b", rf"{NUM}\s*m\b"):
        return None
    length = _num(q, rf"(?:length\s*(?:is|=|:)?\s*|of length\s*)({NUM})\s*m\b")
    if length is None or length <= 0:
        return None
    return PhysicsPlan("physics_pendulum", {"length": length}, _compute_pendulum_period)


def _plan_wave_speed(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bwave\b", r"\bfrequency\b", r"\bwavelength\b", rf"{NUM}\s*hz\b", rf"{NUM}\s*m\b"):
        return None
    frequency = _num(q, rf"({NUM})\s*hz\b")
    wavelength = _num(q, rf"(?:wavelength\s*(?:is|=|:)?\s*)({NUM})\s*m\b")
    if None in (frequency, wavelength):
        return None
    return PhysicsPlan(
        "physics_wave_speed",
        {"frequency": frequency, "wavelength": wavelength},
        _compute_wave_speed,
    )


def _doppler_values(q: str) -> Optional[Tuple[float, float, float]]:
    frequency = _num(q, rf"({NUM})\s*hz\b")
    sound_speed = _num(q, rf"(?:sound speed|speed of sound)\s*(?:is|=|:)?\s*({NUM})\s*m/s")
    if None in (frequency, sound_speed):
        return None
    return frequency, sound_speed


def _plan_doppler_observer(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bdoppler\b", r"\bobserver\s+(?:moves|moving)\b", r"\btoward", rf"{NUM}\s*hz\b", rf"{NUM}\s*m/s"):
        return None
    values = _doppler_values(q)
    if values is None:
        return None
    frequency, sound_speed = values
    observer_speed = _num(q, rf"observer[^\d]{{0,40}}?({NUM})\s*m/s")
    if observer_speed is None:
        return None
    return PhysicsPlan(
        "physics_doppler_observer",
        {"frequency": frequency, "sound_speed": sound_speed, "observer_speed": observer_speed},
        _compute_doppler_observer,
    )


def _plan_doppler_source(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bdoppler\b", r"\bsource\s+(?:moves|moving)\b", r"\btoward", rf"{NUM}\s*hz\b", rf"{NUM}\s*m/s"):
        return None
    values = _doppler_values(q)
    if values is None:
        return None
    frequency, sound_speed = values
    source_speed = _num(q, rf"source[^\d]{{0,40}}?({NUM})\s*m/s")
    if source_speed is None:
        return None
    return PhysicsPlan(
        "physics_doppler_source",
        {"frequency": frequency, "sound_speed": sound_speed, "source_speed": source_speed},
        _compute_doppler_source,
    )


def _plan_snell(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bsnell\b|\brefraction\b|\brefracted angle\b", rf"{NUM}\s*(?:degrees|°)", r"\brefractive index\b"):
        return None
    incident = _num(q, rf"({NUM})\s*(?:degrees|°|\bdeg\b)")
    index = _num(q, rf"refractive index\s*(?:is|=|:)?\s*({NUM})")
    if None in (incident, index) or index <= 0:
        return None
    return PhysicsPlan("physics_snell", {"incident": incident, "index": index}, _compute_snell)


def _plan_lens(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\blens\b", r"\bobject distance\b|\bobject is\b", r"\bfocal length\b", rf"{NUM}\s*m\b"):
        return None
    object_distance = _num(q, rf"(?:object distance\s*(?:is|=|:)?\s*|object is\s*)({NUM})\s*m\b")
    focal = _num(q, rf"focal length\s*(?:is|=|:)?\s*({NUM})\s*m\b")
    if None in (object_distance, focal) or focal <= 0 or object_distance <= 0:
        return None
    return PhysicsPlan(
        "physics_lens",
        {"object_distance": object_distance, "focal": focal},
        _compute_lens,
    )


def _plan_coulomb(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bcoulomb\b|\belectrostatic force\b", r"\bcharges?\b", r"\bc\b", rf"{NUM}\s*m\b"):
        return None
    charges = _nums(q, rf"({NUM})\s*c\b")
    radius = _num(q, rf"(?:separated by|distance of|distance)\s+({NUM})\s*m\b")
    if len(charges) < 2 or radius is None or radius <= 0:
        return None
    return PhysicsPlan(
        "physics_coulomb",
        {"q1": charges[0], "q2": charges[1], "radius": radius},
        _compute_coulomb,
    )


def _plan_electric_potential(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\belectric potential\b", r"\bcharge\b", r"\bc\b", rf"{NUM}\s*m\b"):
        return None
    charge = _num(q, rf"charge\s*(?:of|is|=|:)?\s*({NUM})\s*c\b")
    radius = _num(q, rf"(?:at|distance of|distance)\s+({NUM})\s*m\b")
    if None in (charge, radius) or radius <= 0:
        return None
    return PhysicsPlan(
        "physics_electric_potential",
        {"charge": charge, "radius": radius},
        _compute_electric_potential,
    )


def _plan_electric_field(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\belectric field\b", r"\bcharge\b", r"\bc\b", rf"{NUM}\s*m\b"):
        return None
    charge = _num(q, rf"charge\s*(?:of|is|=|:)?\s*({NUM})\s*c\b")
    radius = _num(q, rf"(?:at|distance of|distance)\s+({NUM})\s*m\b")
    if None in (charge, radius) or radius <= 0:
        return None
    return PhysicsPlan(
        "physics_electric_field",
        {"charge": charge, "radius": radius},
        _compute_electric_field,
    )


def _plan_parallel_resistors(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bresistors?\b", r"\bparallel\b", r"\bequivalent\b|\bcombined\b|\bequivalent resistance\b", rf"{NUM}\s*(?:ohms?|ω)\b"):
        return None
    values = _nums(q, rf"({NUM})\s*(?:ohms?|ω)\b")
    if len(values) < 2 or any(value <= 0 for value in values[:2]):
        return None
    return PhysicsPlan(
        "physics_parallel_resistors",
        {"r1": values[0], "r2": values[1]},
        _compute_parallel_resistors,
    )


def _plan_series_resistors(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bresistors?\b", r"\bseries\b", r"\bequivalent\b|\bcombined\b", rf"{NUM}\s*(?:ohms?|ω)\b"):
        return None
    values = _nums(q, rf"({NUM})\s*(?:ohms?|ω)\b")
    if len(values) < 2:
        return None
    return PhysicsPlan(
        "physics_series_resistors",
        {"r1": values[0], "r2": values[1]},
        _compute_series_resistors,
    )


def _plan_capacitor_energy(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bcapacitor\b", r"\benergy\b|\bstored\b", r"\bf\b", r"\bv\b"):
        return None
    capacitance = _num(q, rf"({NUM})\s*f\b")
    voltage = _num(q, rf"(?:(?:charged to|at|voltage\s*(?:is|=|:)?)\s*)({NUM})\s*v\b")
    if None in (capacitance, voltage):
        return None
    return PhysicsPlan(
        "physics_capacitor_energy",
        {"capacitance": capacitance, "voltage": voltage},
        _compute_capacitor_energy,
    )


def _plan_emf(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bemf\b|\binduced voltage\b|\bfaraday\b", r"\bturns?\b", r"\bt\b", r"\bm\^2\b|\bm²\b", r"\bs\b"):
        return None
    turns = _num(q, rf"({NUM})\s*-?\s*turns?\b")
    area = _num(q, rf"({NUM})\s*m(?:\^2|²)\b")
    fields = _nums(q, rf"({NUM})\s*t\b")
    time = _num(q, rf"(?:in|over|during)\s+({NUM})\s*s(?:econds?)?\b")
    if turns is None or area is None or len(fields) < 2 or time is None or time <= 0:
        return None
    return PhysicsPlan(
        "physics_emf",
        {"turns": int(turns), "area": area, "b1": fields[0], "b2": fields[1], "time": time},
        _compute_emf,
    )


def _plan_rms(q: str) -> Optional[PhysicsPlan]:
    if not re.search(r"\brms\b|\broot mean square\b", q, re.I):
        return None
    kind = None
    unit = None
    if re.search(r"\bcurrent\b|\bamp", q, re.I):
        kind, unit = "I", "A"
    elif re.search(r"\bvoltage\b|\bvolt", q, re.I):
        kind, unit = "V", "V"
    else:
        return None
    peak = _num(q, rf"({NUM})\s*{re.escape(unit)}\b")
    if peak is None:
        return None
    return PhysicsPlan("physics_rms", {"kind": kind, "unit": unit, "peak": peak}, _compute_rms)


def _plan_heat(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bheat\b", r"\bspecific heat\b", r"j/kg/k|j/\(kg k\)|j kg", rf"{NUM}\s*kg", r"\bK\b"):
        return None
    mass = _num(q, rf"({NUM})\s*kg")
    specific_heat = _num(q, rf"specific heat(?: capacity)?\s*(?:is|=|:)?\s*({NUM})\s*j/(?:kg\s*k|kg/k)")
    delta_t = _num(q, rf"(?:temperature rise|rise in temperature|by|of)\s*({NUM})\s*k\b")
    if None in (mass, specific_heat, delta_t):
        return None
    return PhysicsPlan(
        "physics_heat",
        {"mass": mass, "specific_heat": specific_heat, "delta_t": delta_t},
        _compute_heat,
    )


def _plan_entropy(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bentropy change\b", r"\bmoles?\b|\bmol\b", r"\bm\^3\b|\bm³\b"):
        return None
    moles = _num(q, rf"({NUM})\s*mol(?:es)?\b")
    volumes = _nums(q, rf"({NUM})\s*m(?:\^3|³)\b")
    if moles is None or len(volumes) < 2 or volumes[0] <= 0 or volumes[1] <= 0:
        return None
    return PhysicsPlan(
        "physics_entropy",
        {"moles": moles, "v1": volumes[0], "v2": volumes[1]},
        _compute_entropy,
    )


def _plan_photoelectric(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bphotoelectric\b", rf"{NUM}\s*hz\b", r"\bwork function\b"):
        return None
    frequency = _num(q, rf"({NUM})\s*hz\b")
    work_function = _num(q, rf"work function\s*(?:is|=|:)?\s*({NUM})\s*j\b")
    if None in (frequency, work_function):
        return None
    return PhysicsPlan(
        "physics_photoelectric",
        {"frequency": frequency, "work_function": work_function},
        _compute_photoelectric,
    )


def _plan_de_broglie(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bde broglie\b", r"\bmomentum\b|\bkg m/s\b"):
        return None
    momentum = _num(q, rf"(?:momentum\s*(?:is|=|:)?\s*|p\s*=\s*)({NUM})\s*kg\s*m/s")
    if momentum is None or momentum <= 0:
        return None
    return PhysicsPlan("physics_de_broglie", {"momentum": momentum}, _compute_de_broglie)


def _plan_uncertainty(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bposition uncertainty\b", r"\bmomentum uncertainty\b|\buncertainty principle\b", rf"{NUM}\s*m\b"):
        return None
    position = _num(q, rf"position uncertainty\s*(?:of|is|=|:)?\s*({NUM})\s*m\b")
    if position is None or position <= 0:
        return None
    return PhysicsPlan("physics_uncertainty", {"position": position}, _compute_uncertainty)


def _plan_rest_energy(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\brest energy\b|\bmass[-\s]energy\b", rf"{NUM}\s*kg"):
        return None
    mass = _num(q, rf"({NUM})\s*kg")
    if mass is None:
        return None
    return PhysicsPlan("physics_rest_energy", {"mass": mass}, _compute_rest_energy)


def _plan_time_dilation(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\btime dilation\b|\bproper time\b|\btraveller\b|\bclock\b", rf"{NUM}\s*s(?:econds?)?\b", rf"{NUM}\s*m/s"):
        return None
    proper_time = _num(q, rf"(?:proper time|traveller|clock)\s*(?:is|=|:)?\s*({NUM})\s*s(?:econds?)?\b")
    speed = _num(q, rf"(?:at|speed|moving at)\s*({NUM})\s*m/s")
    if None in (proper_time, speed):
        return None
    return PhysicsPlan(
        "physics_time_dilation",
        {"proper_time": proper_time, "speed": speed},
        _compute_time_dilation,
    )


def _plan_decay(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bhalf[-\s]life\b", r"\binitial\b|\bstarts?\s+with\b", rf"{NUM}\s*s(?:econds?)?\b"):
        return None
    initial = _num(q, rf"(?:initial(?:ly)?|starts?\s+with)\s*(?:number\s*)?(?:is|=|:)?\s*({NUM})")
    half_life = _num(q, rf"half[-\s]life\s*(?:of|is|=|:)?\s*({NUM})\s*s(?:econds?)?\b")
    elapsed = _num(q, rf"after\s+({NUM})\s*s(?:econds?)?\b")
    if None in (initial, half_life, elapsed) or half_life <= 0:
        return None
    return PhysicsPlan(
        "physics_decay",
        {"initial": initial, "half_life": half_life, "elapsed": elapsed},
        _compute_decay,
    )


def _plan_hubble(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bhubble\b", r"\bmpc\b", r"km/s/mpc"):
        return None
    distance = _num(q, rf"({NUM})\s*mpc\b")
    hubble = _num(q, rf"(?:hubble constant|h0)\s*(?:is|=|:)?\s*({NUM})\s*km/s/mpc")
    if None in (distance, hubble):
        return None
    return PhysicsPlan("physics_hubble", {"distance": distance, "hubble": hubble}, _compute_hubble)


def _plan_hydrostatic(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bpressure at (?:a )?depth\b|\bhydrostatic pressure\b", r"kg/m\^3|kg/m³", rf"{NUM}\s*m\b"):
        return None
    density = _num(q, rf"({NUM})\s*kg/m(?:\^3|³)\b")
    depth = _num(q, rf"(?:depth|at a depth of)\s*(?:is|=|:)?\s*({NUM})\s*m\b")
    if None in (density, depth):
        return None
    return PhysicsPlan("physics_hydrostatic", {"density": density, "depth": depth}, _compute_hydrostatic)


def _plan_percent_uncertainty(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bpercentage uncertainty\b", r"\bmeasured value\b", r"\babsolute uncertainty\b", r"\d"):
        return None
    measured = _num(q, rf"measured value\s*(?:is|=|:)?\s*({NUM})")
    absolute = _num(q, rf"absolute uncertainty\s*(?:is|=|:)?\s*({NUM})")
    if None in (measured, absolute) or measured <= 0:
        return None
    return PhysicsPlan(
        "physics_percent_uncertainty",
        {"measured": measured, "absolute": absolute},
        _compute_uncertainty_percent,
    )


# ---------------------------------------------------------------------------
# Boundary-aware planners.
#
# The first generation of planners used a "first number wins" shortcut. That
# silently failed whenever a question carried a second medium, a second body,
# a second temperature, an atmospheric offset, or a unit family such as eV or
# cm. These planners derive the actual governing relation instead. They run
# before the narrow legacy planners so the richer physical model wins.
# ---------------------------------------------------------------------------

_WORD_NUMBERS = {"one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0}


def _word_count(q: str) -> int:
    match = re.search(r"\b(one|two|three|four)\b", q, re.I)
    return int(_WORD_NUMBERS[match.group(1).lower()]) if match else 0


def _moles(q: str) -> Optional[float]:
    value = _num(q, rf"({NUM})\s*mol(?:e|es)?\b")
    if value is not None:
        return value
    match = re.search(r"\b(one|two|three|four)\s+mol(?:e|es)?\b", q, re.I)
    return _WORD_NUMBERS.get(match.group(1).lower()) if match else None


def _compute_incline_friction(p: Params) -> str:
    needed = p["mass"] * G0 * math.sin(math.radians(p["angle"]))
    maximum = p["mu"] * p["mass"] * G0 * math.cos(math.radians(p["angle"]))
    if needed <= maximum:
        return (
            f"The required static friction is {_fmt(needed)} N up the incline; "
            f"the maximum available is {_fmt(maximum)} N, so equilibrium is possible."
        )
    return (
        f"The required static friction is {_fmt(needed)} N up the incline, but the "
        f"maximum available is {_fmt(maximum)} N; the block slides down."
    )


def _plan_incline_friction(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q, r"\bincline\b|\bslope\b", r"\bfriction\b", rf"{NUM}\s*kg",
        r"\bcoefficient\b", rf"{NUM}\s*(?:degrees?|°|\bdeg\b)",
    ):
        return None
    mass = _num(q, rf"({NUM})\s*kg")
    angle = _num(q, rf"({NUM})\s*(?:degrees?|°|\bdeg\b)")
    mu = _num(
        q,
        rf"(?:coefficient\s*(?:of\s*)?(?:static\s*)?friction|friction\s*coefficient)"
        rf"[^\d]{{0,24}}?({NUM})",
    )
    if None in (mass, angle, mu):
        return None
    return PhysicsPlan(
        "physics_incline_friction",
        {"mass": mass, "angle": angle, "mu": mu},
        _compute_incline_friction,
    )


def _compute_inelastic_two(p: Params) -> str:
    value = (p["m1"] * p["u1"] + p["m2"] * p["u2"]) / (p["m1"] + p["m2"])
    return _formula(
        "v = (m₁u₁ + m₂u₂)/(m₁ + m₂)",
        f"({_fmt(p['m1'])} × {_fmt(p['u1'])} + {_fmt(p['m2'])} × {_fmt(p['u2'])})/"
        f"({_fmt(p['m1'])} + {_fmt(p['m2'])})",
        value,
        "m/s",
    )


def _plan_inelastic_two(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q, r"\bsticks?\s+(?:together|to)\b|\bperfectly\s+inelastic\b",
        rf"{NUM}\s*kg", rf"{NUM}\s*m/s",
    ):
        return None
    masses = _nums(q, rf"({NUM})\s*kg")
    speeds = _nums(q, rf"({NUM})\s*m/s")
    if len(masses) < 2 or len(speeds) < 2:
        return None
    second_speed = speeds[1]
    if re.search(r"\bopposite\b|\bhead[-\s]on\b|\btoward each other\b", q, re.I):
        second_speed = -abs(second_speed)
    return PhysicsPlan(
        "physics_inelastic_two",
        {"m1": masses[0], "m2": masses[1], "u1": speeds[0], "u2": second_speed},
        _compute_inelastic_two,
    )


def _compute_elastic_general(p: Params) -> str:
    m1, m2, u1, u2 = p["m1"], p["m2"], p["u1"], p["u2"]
    v1 = ((m1 - m2) * u1 + 2 * m2 * u2) / (m1 + m2)
    v2 = ((m2 - m1) * u2 + 2 * m1 * u1) / (m1 + m2)
    return (
        "Using one-dimensional elastic-collision conservation, "
        f"v₁ = {_fmt(v1)} m/s and v₂ = {_fmt(v2)} m/s."
    )


def _plan_elastic_general(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q, r"\belastic\b", r"\bcollis|\bcollid|\bhit\b", rf"{NUM}\s*kg",
        rf"{NUM}\s*m/s", r"\bfinal\b|\bvelocity|\bspeed",
    ):
        return None
    masses = _nums(q, rf"({NUM})\s*kg")
    speeds = _nums(q, rf"({NUM})\s*m/s")
    if not masses or not speeds:
        return None
    first_mass = masses[0]
    if len(masses) >= 2:
        second_mass = masses[1]
    elif re.search(r"\bidentical\b|\bsame mass\b", q, re.I):
        second_mass = first_mass
    else:
        return None
    first_speed = speeds[0]
    second_speed = speeds[1] if len(speeds) >= 2 else 0.0
    if re.search(r"\bopposite\b|\bhead[-\s]on\b|\btoward each other\b", q, re.I):
        second_speed = -abs(second_speed)
    return PhysicsPlan(
        "physics_elastic_general",
        {"m1": first_mass, "m2": second_mass, "u1": first_speed, "u2": second_speed},
        _compute_elastic_general,
    )


def _compute_projectile_elevated(p: Params) -> str:
    angle = math.radians(p["angle"])
    vx = p["speed"] * math.cos(angle)
    vy = p["speed"] * math.sin(angle)
    flight = (vy + math.sqrt(vy * vy + 2 * G0 * p["height"])) / G0
    value = vx * flight
    return (
        f"Horizontal velocity is {_fmt(vx)} m/s. Solving y = {_fmt(p['height'])} m "
        f"+ {_fmt(vy)}t - ½gt² = 0 gives t = {_fmt(flight)} s, so the range is "
        f"{_fmt(value)} m."
    )


def _plan_projectile_elevated(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q, r"\bprojectile\b|\blaunch", r"\brange\b|\bhorizontal distance\b|\bland\b",
        rf"{NUM}\s*m/s", rf"{NUM}\s*(?:degrees|°|\bdeg\b)",
        r"\bcliff\b|\bheight\b|\babove\b",
    ):
        return None
    values = _projectile_values(q)
    height = _num(
        q,
        rf"(?:cliff|height|above)[^\d]{{0,24}}?({NUM})\s*m\b",
    )
    if values is None or height is None:
        return None
    return PhysicsPlan(
        "physics_projectile_elevated",
        {"speed": values[0], "angle": values[1], "height": height},
        _compute_projectile_elevated,
    )


def _volume_values(q: str) -> List[float]:
    values: List[float] = []
    pattern = rf"({NUM})\s*(m\^3|m³|litres?|liters?|l)\b"
    for match in re.finditer(pattern, q, re.I):
        try:
            number = float(match.group(1))
        except ValueError:
            continue
        unit = match.group(2).lower()
        values.append(number if unit.startswith("m") else number * 1e-3)
    return values


def _compute_entropy_general(p: Params) -> str:
    volume_term = p["n"] * R_GAS * math.log(p["v2"] / p["v1"])
    if p.get("cv") is None:
        return (
            f"The volume contribution is {_fmt(volume_term)} J/K. With a temperature "
            "change the full result is nCᵥln(T₂/T₁) + nRln(V₂/V₁); Cᵥ is required "
            "to finish it."
        )
    temperature_term = p["n"] * p["cv"] * math.log(p["t2"] / p["t1"])
    value = temperature_term + volume_term
    return (
        f"ΔS = nCᵥln(T₂/T₁) + nRln(V₂/V₁) = {_fmt(temperature_term)} + "
        f"{_fmt(volume_term)} = {_fmt(value)} J/K."
    )


def _plan_entropy_general(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bentropy change\b|\bdelta s\b", r"\bmol\b|\bmoles\b", r"m\^3|m³|\bl\b|\blitre|\bliter"):
        return None
    moles = _moles(q)
    volumes = _volume_values(q)
    if moles is None or len(volumes) < 2 or volumes[0] <= 0 or volumes[1] <= 0:
        return None
    temperatures = _nums(q, rf"({NUM})\s*k\b")
    cv = None
    match = re.search(
        r"(?:c_?v|specific heat at constant volume)\s*(?:is|=|:)?\s*"
        rf"({NUM})\s*r\s*/\s*({NUM})",
        q,
        re.I,
    )
    if match:
        cv = float(match.group(1)) / float(match.group(2)) * R_GAS
    elif re.search(r"\bmonatomic\b|\bideal gas\b", q, re.I):
        cv = 1.5 * R_GAS
    return PhysicsPlan(
        "physics_entropy_general",
        {
            "n": moles,
            "v1": volumes[0],
            "v2": volumes[1],
            "t1": temperatures[0] if len(temperatures) >= 2 else 1.0,
            "t2": temperatures[1] if len(temperatures) >= 2 else 1.0,
            "cv": cv,
        },
        _compute_entropy_general,
    )


def _compute_heat_change(p: Params) -> str:
    change = p["mass"] * p["specific_heat"] * (p["final"] - p["initial"])
    if p.get("removed"):
        return (
            f"Q_system = mc(T₂ − T₁) = {_fmt(change)} J, so the heat removed is "
            f"{_fmt(abs(change))} J."
        )
    return _formula(
        "Q = mc(T₂ − T₁)",
        f"{_fmt(p['mass'])} × {_fmt(p['specific_heat'])} × "
        f"({_fmt(p['final'])} − {_fmt(p['initial'])})",
        change,
        "J",
    )


def _plan_heat_change(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q, r"\bheat\b|\bcools?\b|\bheats?\b|\btemperature\b", r"\bspecific heat\b",
        rf"{NUM}\s*kg", r"j/\(?kg\s*k\)?", rf"{NUM}\s*k\b", r"\bto\b",
    ):
        return None
    mass = _num(q, rf"({NUM})\s*kg")
    specific_heat = _num(
        q,
        rf"specific heat(?: capacity)?\s*(?:is|=|:)?\s*({NUM})\s*j/\(?kg\s*k\)?",
    )
    temperatures = _nums(q, rf"({NUM})\s*k\b")
    if None in (mass, specific_heat) or len(temperatures) < 2:
        return None
    return PhysicsPlan(
        "physics_heat_change",
        {
            "mass": mass,
            "specific_heat": specific_heat,
            "initial": temperatures[0],
            "final": temperatures[1],
            "removed": bool(re.search(r"\bremoved\b|\bcools?\b", q, re.I)),
        },
        _compute_heat_change,
    )


def _plan_adiabatic_work(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q, r"\badiabatic(?:ally)?\b", r"\bmonatomic\b|\bideal gas\b", r"\bmole\b|\bmoles\b",
        rf"{NUM}\s*k\b", r"\bfrom\b.*\bto\b",
    ):
        return None
    n = _moles(q)
    temperatures = _nums(q, rf"({NUM})\s*k\b")
    if n is None or len(temperatures) < 2:
        return None
    work = n * 1.5 * R_GAS * (temperatures[0] - temperatures[1])
    text = (
        f"For a reversible monatomic ideal-gas expansion, W = nCᵥ(T₁ − T₂) = "
        f"{_fmt(n)} × {_fmt(1.5 * R_GAS)} × "
        f"({_fmt(temperatures[0])} − {_fmt(temperatures[1])}) = {_fmt(work)} J."
    )
    return PhysicsPlan("physics_adiabatic_work", {}, lambda _: text)


def _compute_parallel_n(p: Params) -> str:
    value = 1.0 / sum(1.0 / resistance for resistance in p["values"])
    substitution = " + ".join(f"1/{_fmt(resistance)}" for resistance in p["values"])
    return _formula("1/R = Σ(1/Rᵢ)", substitution, value, "Ω")


def _plan_parallel_n(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bresistors?\b", r"\bparallel\b", r"\bequivalent\b|\bcombined\b"):
        return None
    values = _nums(q, rf"({NUM})\s*(?:ohms?|ω)\b")
    count = _word_count(q)
    if len(values) == 1 and count > 1:
        values = values * count
    if len(values) < 2 or any(value <= 0 for value in values):
        return None
    return PhysicsPlan("physics_parallel_n", {"values": values}, _compute_parallel_n)


def _plan_rotating_emf(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q, r"\bemf\b|\binduced\b|\bfaraday\b", r"\bcoil\b", r"\bturns?\b",
        r"m\^2|m²", r"\bt\b", r"\bs\b", r"\brotat|\bnormal\b|\bperpendicular\b",
    ):
        return None
    turns = _num(q, rf"({NUM})\s*-?\s*turns?\b")
    area = _num(q, rf"({NUM})\s*m(?:\^2|²)\b")
    field = _num(q, rf"({NUM})\s*t\b")
    time = _num(q, rf"(?:in|over|during)\s+({NUM})\s*s(?:econds?)?\b")
    if None in (turns, area, field, time) or time <= 0:
        return None
    value = turns * area * field / time
    text = (
        f"The flux changes by N B A = {_fmt(turns)} × {_fmt(field)} × {_fmt(area)} "
        f"Wb-turns, so the mean emf is {_fmt(value)} V."
    )
    return PhysicsPlan("physics_rotating_emf", {}, lambda _: text)


def _plan_coulomb_coords(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bcoulomb\b|\belectrostatic force\b", r"\bcharges?\b", r"\bc\b", r"\("):
        return None
    charges = _nums(q, rf"({NUM})\s*c\b")
    coords = [
        (float(pair[0]), float(pair[1]))
        for pair in re.findall(
            rf"\(\s*({NUM})\s*(?:m)?\s*,\s*({NUM})\s*(?:m)?\s*\)", q
        )
    ]
    if len(charges) < 2 or len(coords) < 2:
        return None
    separation = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
    if separation <= 0:
        return None
    value = K_E * abs(charges[0] * charges[1]) / separation ** 2
    text = (
        f"Separation is √(({_fmt(coords[1][0])} − {_fmt(coords[0][0])})² + "
        f"({_fmt(coords[1][1])} − {_fmt(coords[0][1])})²) = {_fmt(separation)} m, so "
        f"F = k|q₁q₂|/r² = {_fmt(value, 3)} N."
    )
    return PhysicsPlan("physics_coulomb_coords", {}, lambda _: text)


def _plan_capacitor_discharge(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bcapacitor\b", r"\breleased\b|\bdischarg", r"\bf\b", r"\bv\b"):
        return None
    capacitance = _num(q, rf"({NUM})\s*f\b")
    voltages = _nums(q, rf"({NUM})\s*v\b")
    if capacitance is None or len(voltages) < 2:
        return None
    high = max(abs(voltage) for voltage in voltages)
    low = abs(voltages[-1])
    value = 0.5 * capacitance * (high ** 2 - low ** 2)
    text = (
        f"Energy released is ½C(V₁² − V₂²) = 1/2 × {_fmt(capacitance)} × "
        f"({_fmt(high)}² − {_fmt(low)}²) = {_fmt(value)} J."
    )
    return PhysicsPlan("physics_capacitor_discharge", {}, lambda _: text)


def _plan_lens_signed(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\blens\b", r"\bfocal length\b", r"\bobject\b", r"\bimage\b", r"\bcm\b|\bm\b"):
        return None
    focal_match = re.search(rf"focal length\s*(?:is|=|:)?\s*({NUM})\s*(cm|m)\b", q, re.I)
    object_match = re.search(
        rf"(?:object is|object distance\s*(?:is|=|:)?|object\s+(?:placed|at|located))"
        rf"\s*({NUM})\s*(cm|m)\b",
        q,
        re.I,
    )
    if not focal_match or not object_match:
        return None
    focal = float(focal_match.group(1)) * (0.01 if focal_match.group(2).lower() == "cm" else 1.0)
    object_distance = float(object_match.group(1)) * (
        0.01 if object_match.group(2).lower() == "cm" else 1.0
    )
    if focal == 0 or object_distance == 0:
        return None
    denominator = 1.0 / focal - 1.0 / object_distance
    if abs(denominator) < 1e-15:
        return PhysicsPlan(
            "physics_lens_signed",
            {},
            lambda _: "The thin-lens equation puts the image at infinity.",
        )
    image = 1.0 / denominator
    kind = "virtual" if image < 0 else "real"
    text = (
        f"Using 1/f = 1/u + 1/v with signed distances, 1/v = 1/({_fmt(focal)}) − "
        f"1/({_fmt(object_distance)}), so v = {_fmt(image)} m, a {kind} image."
    )
    return PhysicsPlan("physics_lens_signed", {}, lambda _: text)


def _plan_doppler_general(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q, r"\bdoppler\b|\bfrequency is heard\b", rf"{NUM}\s*hz\b",
        r"\bsound speed\b|\bspeed of sound\b", rf"{NUM}\s*m/s",
    ):
        return None
    frequency = _num(q, rf"({NUM})\s*hz\b")
    sound_speed = _num(
        q, rf"(?:sound speed|speed of sound)\s*(?:is|=|:)?\s*({NUM})\s*m/s"
    )
    speed = _num(q, rf"(?:moves|moving|travels?)[^\d]{{0,48}}?({NUM})\s*m/s")
    if None in (frequency, sound_speed, speed):
        return None
    away = bool(re.search(r"\baway\b|\breced", q, re.I))
    observer_moves = bool(
        re.search(r"\bobserver\b[^.]{0,40}\b(?:moves|moving|travels?)\b", q, re.I)
    )
    source_moves = bool(
        re.search(r"\bsource\b[^.]{0,40}\b(?:moves|moving|travels?)\b", q, re.I)
    )
    # The word "source" alone is not enough: the moving body decides the
    # Doppler branch. "Source emits ... observer moves toward it" is observer.
    source = source_moves or (not observer_moves and bool(re.search(r"\bsource\b", q, re.I)))
    if source:
        denominator = sound_speed + speed if away else sound_speed - speed
        if denominator <= 0:
            return PhysicsPlan(
                "physics_doppler_general",
                {},
                lambda _: "The source speed reaches the wave speed, so the simple Doppler formula has no finite value here.",
            )
        value = frequency * sound_speed / denominator
        label = "fv/(v + vₛ)" if away else "fv/(v − vₛ)"
    else:
        factor = sound_speed - speed if away else sound_speed + speed
        value = frequency * factor / sound_speed
        label = "f(v − v₀)/v" if away else "f(v + v₀)/v"
    return PhysicsPlan(
        "physics_doppler_general",
        {},
        lambda _: _formula(f"f' = {label}", f"{_fmt(frequency)} × ...", value, "Hz"),
    )


def _plan_grating(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q, r"\bgrating\b", r"\blines?\s*(?:per|/)\s*mm\b", rf"{NUM}\s*nm\b",
        r"\bangle\b|\bmaximum\b|\border\b",
    ):
        return None
    lines = _num(q, rf"({NUM})\s*lines?\s*(?:per|/)\s*mm\b")
    wavelength_nm = _num(q, rf"({NUM})\s*nm\b")
    if None in (lines, wavelength_nm) or lines <= 0:
        return None
    order = 1
    if re.search(r"\bsecond[-\s]order\b|\bm\s*=\s*2\b", q, re.I):
        order = 2
    elif re.search(r"\bthird[-\s]order\b|\bm\s*=\s*3\b", q, re.I):
        order = 3
    spacing = 1e-3 / lines
    sine = order * wavelength_nm * 1e-9 / spacing
    if abs(sine) > 1:
        text = (
            f"Grating spacing is {_fmt(spacing)} m, but order {order} would need "
            f"sin θ = {_fmt(sine)} > 1, so that order does not exist."
        )
    else:
        angle = math.degrees(math.asin(sine))
        text = (
            f"Grating spacing is d = 1/({_fmt(lines)} lines/mm) = {_fmt(spacing)} m. "
            f"For order {order}, sin θ = mλ/d = {_fmt(sine)}, so θ = {_fmt(angle)}°."
        )
    return PhysicsPlan("physics_grating", {}, lambda _: text)


def _plan_photoelectric_ev(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q,
        r"\bphotoelectric\b|\bwork function\b.*\bemit",
        rf"{NUM}\s*hz\b",
        r"\bwork function\b",
        r"\bev\b",
    ):
        return None
    frequency = _num(q, rf"({NUM})\s*hz\b")
    work_ev = _num(q, rf"work function\s*(?:is|=|:)?\s*({NUM})\s*ev\b")
    if None in (frequency, work_ev):
        return None
    photon_energy = H_PLANCK * frequency
    value = max(0.0, photon_energy - work_ev * E_CHARGE)
    text = (
        f"Work function = {_fmt(work_ev)} eV = {_fmt(work_ev * E_CHARGE)} J and hf = "
        f"{_fmt(photon_energy)} J, so Kmax = {_fmt(value, 3)} J."
    )
    return PhysicsPlan("physics_photoelectric_ev", {}, lambda _: text)


def _plan_de_broglie_energy(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bde broglie\b", r"\bkinetic energy\b", r"\bev\b", r"\belectron\b"):
        return None
    energy_ev = _num(q, rf"({NUM})\s*ev\b")
    if energy_ev is None or energy_ev <= 0:
        return None
    momentum = math.sqrt(2 * M_ELECTRON * energy_ev * E_CHARGE)
    wavelength = H_PLANCK / momentum
    text = (
        f"For a non-relativistic electron, p = √(2mE) with E = {_fmt(energy_ev)} eV, "
        f"so λ = h/p = {_fmt(wavelength, 3)} m."
    )
    return PhysicsPlan("physics_de_broglie_energy", {}, lambda _: text)


def _plan_infinite_well(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q, r"\binfinite (?:square )?well\b", r"\bnm\b", r"\bn\s*=\s*\d+", r"\bwavelength\b"
    ):
        return None
    width_nm = _num(q, rf"({NUM})\s*nm\b")
    levels = [int(float(value)) for value in _nums(q, r"\bn\s*=\s*(\d+)\b")]
    if width_nm is None or len(levels) < 2:
        return None
    high, low = max(levels), min(levels)
    width = width_nm * 1e-9
    energy = (high ** 2 - low ** 2) * H_PLANCK ** 2 / (8 * M_ELECTRON * width ** 2)
    wavelength = H_PLANCK * C_LIGHT / energy
    text = (
        f"ΔE = (nᵢ² − n_f²)h²/(8mL²) = {_fmt(energy, 3)} J for n={high}→{low}, so "
        f"λ = hc/ΔE = {_fmt(wavelength, 3)} m."
    )
    return PhysicsPlan("physics_infinite_well", {}, lambda _: text)


_SPIN_AXIS_LABEL = {"x": "Sₓ", "y": "Sᵧ", "z": "S_z"}

_SPIN_PREPARED_AXIS_PATTERNS = (
    r"spin[-\s]?up\s+(?:state\s+)?along\s+(?:the\s+)?([xyz])\b",
    r"prepared\s+(?:in\s+)?(?:the\s+)?spin[-\s]?up\s+(?:state\s+)?(?:along|in)\s+(?:the\s+)?([xyz])\b",
    r"spin[-\s]?up\s+in\s+the\s+([xyz])\s+direction",
    r"S_?([xyz])\s*[-–]?\s*up\b",
)

_SPIN_MEASURED_AXIS_PATTERNS = (
    r"spin\s+along\s+(?:the\s+)?([xyz])\s+(?:axis\s+)?is\s+measur",
    r"measur\w*\s+(?:the\s+)?spin\s+along\s+(?:the\s+)?([xyz])\b",
    r"measur\w*\s+S_?([xyz])\b",
    r"measur\w*\s+along\s+(?:the\s+)?([xyz])\b",
    r"measur\w*\s+(?:the\s+)?([xyz])\s+(?:spin\s+)?(?:axis|component)\b",
)


def _spin_axis(q: str, patterns) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, q, re.I)
        if match:
            return match.group(1).lower()
    return None


def _compute_spin_measurement(params: Params) -> str:
    prepared = str(params.get("prepared") or "").lower()
    measured = str(params.get("measured") or "").lower()
    if prepared and measured:
        axis = _SPIN_AXIS_LABEL.get(measured, f"S_{measured}")
        if prepared == measured:
            return (
                f"For a spin-1/2 particle prepared spin-up along {prepared}, measuring "
                f"{axis} along the same axis gives +ℏ/2 with certainty (probability 1); "
                f"−ℏ/2 has probability 0."
            )
        return (
            f"For a spin-1/2 particle prepared spin-up along {prepared}, measuring the "
            f"orthogonal {axis} component gives +ℏ/2 or −ℏ/2 with equal probability "
            f"1/2 each (50% each); the {prepared}-up state is an equal superposition of "
            f"{axis} eigenstates."
        )
    return (
        "For a spin-1/2 particle, measuring along an axis orthogonal to the prepared "
        "spin-up axis gives +ℏ/2 or −ℏ/2 with equal probability 1/2 each (50% each); "
        "measuring along the same axis gives +ℏ/2 with certainty."
    )


def _plan_spin_measurement(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q,
        r"\bspin[-\s]?(?:1/2|one[-\s]half)\b",
        r"\bmeasur",
        r"\balong\s+[xyz]\b|\bS_?[xyz]\b|\b[xyz]\s+(?:spin\s+)?(?:axis|component)\b",
    ):
        return None
    return PhysicsPlan(
        "physics_spin",
        {
            "prepared": _spin_axis(q, _SPIN_PREPARED_AXIS_PATTERNS),
            "measured": _spin_axis(q, _SPIN_MEASURED_AXIS_PATTERNS),
        },
        _compute_spin_measurement,
    )


def _plan_length_contraction(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q, r"\bproper length\b|\blength contraction\b|\brod\b", r"\d\s*c\b|\bc\b",
        r"\bobserver\b|\bmeasure\b",
    ):
        return None
    proper = _num(q, rf"proper length\s*(?:of|is|=|:)?\s*({NUM})\s*m\b")
    if proper is None:
        proper = _num(q, rf"({NUM})\s*m\b")
    beta = _num(q, rf"({NUM})\s*c\b")
    if None in (proper, beta) or beta <= 0 or beta >= 1:
        return None
    gamma = 1.0 / math.sqrt(1.0 - beta ** 2)
    length = proper / gamma
    text = (
        f"γ = 1/√(1 − {_fmt(beta)}²) = {_fmt(gamma)}, so the observer measures "
        f"L = L₀/γ = {_fmt(length)} m."
    )
    return PhysicsPlan("physics_length_contraction", {}, lambda _: text)


def _plan_velocity_addition(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\brelative to\b|\bvelocity addition\b", r"\bforwards?\b|\bprobe\b|\blaunch", r"\d\s*c\b|\bc\b"):
        return None
    betas = _nums(q, rf"({NUM})\s*c\b")
    if len(betas) < 2:
        return None
    total = (betas[0] + betas[1]) / (1.0 + betas[0] * betas[1])
    text = (
        f"Relativistic addition gives u = ({_fmt(betas[0])} + {_fmt(betas[1])})/"
        f"(1 + {_fmt(betas[0])}×{_fmt(betas[1])})c = {_fmt(total)}c."
    )
    return PhysicsPlan("physics_velocity_addition", {}, lambda _: text)


def _plan_redshift(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bredshift\b|\bz\s*=", r"\bhubble\b", r"km/s/mpc"):
        return None
    z = _num(q, rf"(?:redshift\s*(?:z)?\s*(?:is|=|:)?\s*|z\s*=\s*)({NUM})")
    hubble = _num(q, rf"(?:hubble constant|h0)\s*(?:is|=|:)?\s*({NUM})\s*km/s/mpc")
    if None in (z, hubble) or hubble <= 0:
        return None
    speed = z * 3.0e5
    distance = speed / hubble
    text = (
        f"Using v ≈ cz gives {_fmt(speed)} km/s. With H₀ = {_fmt(hubble)} km/s/Mpc, "
        f"d = v/H₀ = {_fmt(distance)} Mpc."
    )
    return PhysicsPlan("physics_redshift", {}, lambda _: text)


def _plan_schwarzschild_rate(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q, r"\bblack hole\b|\bschwarzschild\b", r"\bclock\b|\brate\b", r"r_?s|schwarzschild radius"
    ):
        return None
    factor = _num(q, rf"r\s*(?:is|=|:)?\s*({NUM})\s*r_?s\b")
    if factor is None or factor <= 1:
        return None
    ratio = math.sqrt(1.0 - 1.0 / factor)
    text = (
        f"For a static clock at r = {_fmt(factor)}Rₛ, the rate relative to infinity is "
        f"√(1 − Rₛ/r) = √(1 − 1/{_fmt(factor)}) = {_fmt(ratio)}."
    )
    return PhysicsPlan("physics_schwarzschild_rate", {}, lambda _: text)


def _plan_hydrostatic_absolute(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q, r"\bpressure at (?:a )?depth\b|\bhydrostatic pressure\b", r"\babsolute\b",
        r"kg/m\^3|kg/m³", rf"{NUM}\s*m\b", r"\bpa\b",
    ):
        return None
    density = _num(q, rf"({NUM})\s*kg/m(?:\^3|³)\b")
    depth = _num(q, rf"(?:depth|at a depth of)\s*(?:is|=|:)?\s*({NUM})\s*m\b")
    atmospheric = _num(q, rf"atmospheric pressure\s*(?:is|=|:)?\s*({NUM})\s*pa\b")
    if None in (density, depth, atmospheric):
        return None
    value = density * G0 * depth + atmospheric
    return PhysicsPlan(
        "physics_hydrostatic_absolute",
        {},
        lambda _: _formula(
            "P_abs = ρgh + P_atm",
            f"{_fmt(density)} × {_fmt(G0)} × {_fmt(depth)} + {_fmt(atmospheric)}",
            value,
            "Pa",
        ),
    )


def _plan_bernoulli_pressure(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q, r"\bbernoulli\b|\bpressure drop\b|\bnarrows\b", r"\bdensity\b",
        r"kg/m\^3|kg/m³", rf"{NUM}\s*m/s",
    ):
        return None
    density = _num(q, rf"({NUM})\s*kg/m(?:\^3|³)\b")
    speeds = _nums(q, rf"({NUM})\s*m/s")
    if density is None or len(speeds) < 2:
        return None
    drop = 0.5 * density * (speeds[1] ** 2 - speeds[0] ** 2)
    return PhysicsPlan(
        "physics_bernoulli",
        {},
        lambda _: _formula(
            "ΔP = ½ρ(v₂² − v₁²)",
            f"1/2 × {_fmt(density)} × ({_fmt(speeds[1])}² − {_fmt(speeds[0])}²)",
            drop,
            "Pa",
        ),
    )


def _plan_debye(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bdebye length\b", r"m\^-3|\bm\^\(-3\)", r"\bev\b"):
        return None
    density = _num(q, rf"({NUM})\s*m\^-3\b")
    temp_ev = _num(q, rf"({NUM})\s*ev\b")
    if None in (density, temp_ev) or density <= 0 or temp_ev <= 0:
        return None
    value = math.sqrt(EPS0 * temp_ev * E_CHARGE / (density * E_CHARGE ** 2))
    text = (
        f"Using λ_D = √(ε₀kT/(ne²)), √({_fmt(EPS0)} × {_fmt(temp_ev)} eV / "
        f"{_fmt(density)} m⁻³) = {_fmt(value, 3)} m."
    )
    return PhysicsPlan("physics_debye", {}, lambda _: text)


def _plan_poisson(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q, r"\bpoisson'?s ratio\b", r"\bp[-\s]wave\b|\bcompressional\b",
        r"\bs[-\s]wave\b|\bshear\b", r"\bkm/s\b|\bm/s\b",
    ):
        return None
    speeds = _nums(q, rf"({NUM})\s*(?:km/s|m/s)\b")
    if len(speeds) < 2:
        return None
    p2, s2 = speeds[0] ** 2, speeds[1] ** 2
    value = (p2 - 2 * s2) / (2 * (p2 - s2))
    return PhysicsPlan(
        "physics_poisson",
        {},
        lambda _: _formula(
            "ν = (vₚ² − 2vₛ²)/(2(vₚ² − vₛ²))",
            f"({_fmt(speeds[0])}² − 2×{_fmt(speeds[1])}²)/(2×({_fmt(speeds[0])}² − {_fmt(speeds[1])}²))",
            value,
            "",
        ).replace("  .", "."),
    )


def _plan_cfl(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q, r"\bexplicit\b|\bfinite[-\s]difference\b", r"\bheat equation\b",
        r"\bstability\b", r"\btime step\b|\bdelta t\b|\bΔt\b",
    ):
        return None
    text = (
        "For the explicit one-dimensional heat equation, the stability limit is "
        "Δt ≤ Δx²/(2D). Larger Δx or smaller D relaxes the bound; violating it "
        "produces growing numerical oscillations."
    )
    return PhysicsPlan("physics_cfl", {}, lambda _: text)


def _plan_monte_carlo_scaling(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q, r"\bmonte carlo\b", r"\bstandard error\b|\berror\b",
        r"1/sqrt\(n\)|1/√n|sqrt\(n\)", r"\bfactor\b|\breduc",
    ):
        return None
    factor = _num(q, rf"reduce[^\d]{{0,40}}?({NUM})")
    if factor is None:
        factor = _num(q, rf"by a factor of\s*({NUM})")
    if factor is None or factor <= 0:
        return None
    samples = factor ** 2
    text = (
        f"Standard error scales as N^(−1/2), so a {_fmt(factor)}× reduction needs "
        f"{_fmt(samples)}× as many samples."
    )
    return PhysicsPlan("physics_monte_carlo", {}, lambda _: text)


def _plan_lyapunov_growth(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\blyapunov\b", r"s\^-1|/s\b", r"\bafter\b"):
        return None
    exponent = _num(q, rf"({NUM})\s*s\^-1")
    elapsed = _num(q, rf"after\s+({NUM})\s*s(?:econds?)?\b")
    if None in (exponent, elapsed):
        return None
    factor = math.exp(exponent * elapsed)
    text = (
        f"Separation grows as exp(λt) = exp({_fmt(exponent)} × {_fmt(elapsed)}) = "
        f"exp({_fmt(exponent * elapsed)}) ≈ {_fmt(factor)}."
    )
    return PhysicsPlan("physics_lyapunov", {}, lambda _: text)


def _plan_nernst(q: str) -> Optional[PhysicsPlan]:
    if not _all(q, r"\bnernst\b", r"\bmonovalent\b|\bz\s*=\s*1\b", r"\bmm\b", r"\bk\b"):
        return None
    temperature = _num(q, rf"({NUM})\s*k\b")
    concentrations = _nums(q, rf"({NUM})\s*mm\b")
    if temperature is None or len(concentrations) < 2 or concentrations[1] <= 0:
        return None
    value = R_GAS * temperature / FARADAY * math.log(concentrations[0] / concentrations[1]) * 1000
    return PhysicsPlan(
        "physics_nernst",
        {},
        lambda _: _formula(
            "E = (RT/zF) ln(c_out/c_in)",
            f"({_fmt(R_GAS)} × {_fmt(temperature)}/{_fmt(FARADAY)}) × "
            f"ln({_fmt(concentrations[0])}/{_fmt(concentrations[1])})",
            value,
            "mV",
        ),
    )


def _plan_snell_two_media(q: str) -> Optional[PhysicsPlan]:
    if not _all(
        q, r"\bsnell\b|\brefraction\b|\brefracted angle\b",
        r"\brefractive index\b", rf"{NUM}\s*(?:degrees|°|\bdeg\b)",
    ):
        return None
    indices = _nums(q, rf"refractive index\s*(?:is|=|:)?\s*({NUM})")
    incident = _num(q, rf"({NUM})\s*(?:degrees|°|\bdeg\b)")
    if len(indices) < 2 or incident is None or indices[1] <= 0:
        return None
    sine = indices[0] * math.sin(math.radians(incident)) / indices[1]
    if abs(sine) > 1:
        text = (
            f"sin θ₂ = n₁sinθ₁/n₂ = {_fmt(indices[0])}×sin({_fmt(incident)}°)/"
            f"{_fmt(indices[1])} = {_fmt(sine)} > 1, so total internal reflection "
            "occurs and there is no refracted ray."
        )
    else:
        angle = math.degrees(math.asin(sine))
        text = (
            f"sin θ₂ = n₁sinθ₁/n₂ = {_fmt(indices[0])}×sin({_fmt(incident)}°)/"
            f"{_fmt(indices[1])} = {_fmt(sine)}, so θ₂ = {_fmt(angle)}°."
        )
    return PhysicsPlan("physics_snell_two_media", {}, lambda _: text)


PHYSICS_CONCEPT_ANSWERS: Sequence = (
    (
        re.compile(r"\bspin[-\s]?(?:1/2|one[-\s]half)\b.*\balong\s+[xyz]\b", re.I),
        _compute_spin_measurement({}),
    ),
    (
        re.compile(r"\blength contraction\b|\bproper length\b.*\bc\b", re.I),
        "Length contraction is L = L₀/γ with γ = 1/√(1 − v²/c²); the moving length is shorter only along the direction of motion.",
    ),
    (
        re.compile(r"\bvelocity addition\b|\brelative to\b.*\bprobe\b", re.I),
        "Collinear relativistic velocities combine as u = (u′ + v)/(1 + u′v/c²), which never exceeds c.",
    ),
    (
        re.compile(r"\bdebye length\b", re.I),
        "The Debye length is λ_D = √(ε₀kT/(ne²)); it sets the distance over which charge imbalance is screened in a plasma.",
    ),
    (
        re.compile(r"\bpoisson'?s ratio\b", re.I),
        "For an isotropic solid, Poisson's ratio is ν = (vₚ² − 2vₛ²)/(2(vₚ² − vₛ²)), where vₚ and vₛ are compressional and shear speeds.",
    ),
    (
        re.compile(
            r"\bbernoulli(?:'s|s)?\s+(?:equation|principle|theorem)\b"
            r"|\bbernoulli\s+flow\b"
            r"|\bpressure drop\b.*\bpipe\b",
            re.I,
        ),
        "For horizontal incompressible flow, Bernoulli gives ΔP = ½ρ(v₂² − v₁²) between two speeds.",
    ),
    (
        re.compile(r"\bnernst\b", re.I),
        "The Nernst potential is E = (RT/zF) ln(c_out/c_in), giving the voltage that balances diffusion for ion charge z.",
    ),
    (
        re.compile(r"\blyapunov\b", re.I),
        "A largest Lyapunov exponent λ means an initial separation grows roughly as exp(λt), so positive λ signals chaotic sensitivity.",
    ),
    (
        re.compile(r"\bmonte carlo\b", re.I),
        "Monte Carlo standard error scales as σ/√N, so reducing error by a factor k needs k² times as many samples.",
    ),
    (
        re.compile(r"\bheat equation\b.*\bstability\b|\bcfl\b", re.I),
        "The explicit 1D heat scheme is stable when Δt ≤ Δx²/(2D); this is the diffusive Courant-type limit.",
    ),
    (
        re.compile(r"\binfinite (?:square )?well\b", re.I),
        "For an infinite well of width L, energies are Eₙ = n²h²/(8mL²), and transition photons have λ = hc/ΔE.",
    ),
    (
        re.compile(r"\bgrating\b", re.I),
        "A diffraction grating obeys d sinθ = mλ, with spacing d = 1/(lines per unit length).",
    ),
    (
        re.compile(r"\bblack hole\b|\bschwarzschild\b", re.I),
        "A static clock at radius r near a Schwarzschild black hole runs slower by √(1 − Rₛ/r) relative to a distant clock.",
    ),
    (
        re.compile(r"\bredshift\b", re.I),
        "At small redshift z, recession speed is v ≈ cz and Hubble distance is d ≈ v/H₀.",
    ),
)


def concept_answer(question: str) -> Optional[str]:
    """Narrow, formula-first answers for applied physics topics without a full numeric parse."""
    text = str(question or "")
    for pattern, answer_text in PHYSICS_CONCEPT_ANSWERS:
        if pattern.search(text):
            return answer_text
    return None


_PLANNERS: Sequence = (
    _plan_snell_two_media,
    _plan_incline_friction,
    _plan_inelastic_two,
    _plan_elastic_general,
    _plan_projectile_elevated,
    _plan_entropy_general,
    _plan_heat_change,
    _plan_adiabatic_work,
    _plan_parallel_n,
    _plan_rotating_emf,
    _plan_coulomb_coords,
    _plan_capacitor_discharge,
    _plan_lens_signed,
    _plan_doppler_general,
    _plan_grating,
    _plan_photoelectric_ev,
    _plan_de_broglie_energy,
    _plan_infinite_well,
    _plan_spin_measurement,
    _plan_length_contraction,
    _plan_velocity_addition,
    _plan_redshift,
    _plan_schwarzschild_rate,
    _plan_hydrostatic_absolute,
    _plan_bernoulli_pressure,
    _plan_debye,
    _plan_poisson,
    _plan_cfl,
    _plan_monte_carlo_scaling,
    _plan_lyapunov_growth,
    _plan_nernst,
    _plan_friction,
    _plan_momentum,
    _plan_impulse,
    _plan_elastic_collision,
    _plan_inelastic_collision,
    _plan_centripetal,
    _plan_orbital_speed,
    _plan_gravity_force,
    _plan_gravity_energy,
    _plan_projectile_range,
    _plan_projectile_height,
    _plan_projectile_time,
    _plan_spring_period,
    _plan_pendulum_period,
    _plan_wave_speed,
    _plan_doppler_observer,
    _plan_doppler_source,
    _plan_snell,
    _plan_lens,
    _plan_coulomb,
    _plan_electric_potential,
    _plan_electric_field,
    _plan_parallel_resistors,
    _plan_series_resistors,
    _plan_capacitor_energy,
    _plan_emf,
    _plan_rms,
    _plan_heat,
    _plan_entropy,
    _plan_photoelectric,
    _plan_de_broglie,
    _plan_uncertainty,
    _plan_rest_energy,
    _plan_time_dilation,
    _plan_decay,
    _plan_hubble,
    _plan_hydrostatic,
    _plan_percent_uncertainty,
)


def plan(question: str) -> Optional[PhysicsPlan]:
    q = _normalize(question)
    for planner in _PLANNERS:
        found = planner(q)
        if found:
            return found
    return None


def answer(question: str) -> Optional[str]:
    found = plan(question)
    if not found:
        return None
    return found.compute(found.params)


__all__ = ["PhysicsPlan", "answer", "concept_answer", "plan"]
