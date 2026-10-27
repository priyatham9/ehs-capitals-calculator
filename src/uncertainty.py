"""
Interval and seeded Monte Carlo propagation of assumed input ranges.

Standard library only. This complements :mod:`montecarlo` (numpy, shaped
distributions) with two things that need no distributional assumption or
third-party package:

  interval_bounds()   exact lower and upper bounds on payback, ROI and net ROI
                      given a [low, high] box on every input. Every input
                      appears exactly once in the model's formulas and each
                      output is monotone in each input, so evaluating the box
                      endpoints gives the tight bounds, not a conservative
                      over-estimate. The bounds hold for ANY joint behaviour of
                      the inputs inside the box, including the correlated
                      behaviour that independent sampling misses.

  simulate()          seeded uniform sampling over the same box using
                      ``random.Random(seed)``: percentiles and the probability
                      that payback is within a hurdle. The same seed and sample
                      count reproduce the same numbers exactly.

THE RANGES ARE ASSUMPTIONS
--------------------------
The model's only parameters are the cost lines, capex, risk reduction and the
horizon. Nothing in this repository measures how uncertain any of them is.
Ranges come from :func:`sensitivity.relative_ranges` (one relative half-width
applied to every input, a placeholder) or from explicit user overrides. Output
bounds describe those assumed ranges, not the world. Uniform sampling also
treats every value in a range as equally plausible and samples inputs
independently, which understates joint spread; the interval bounds do not
depend on either choice.
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from capitals import evaluate
from sensitivity import COST_PREFIX, Scenario, relative_ranges

DEFAULT_SEED = 1904
DEFAULT_SAMPLES = 20_000

Ranges = Dict[str, Tuple[float, float]]


def build_ranges(
    scenario: Scenario,
    spread: float,
    overrides: Optional[Ranges] = None,
    vary_horizon: bool = False,
) -> Ranges:
    """Assumed input box: a uniform relative band, then explicit overrides.

    Raises:
        ValueError: on an unknown parameter, an inverted range, or a range that
            the model would reject (capex <= 0, risk reduction outside [0, 1],
            negative cost, horizon < 1).
    """
    ranges = relative_ranges(scenario, fraction=spread, include_horizon=vary_horizon)
    names = set(scenario.parameter_names())
    for name, (low, high) in (overrides or {}).items():
        if name not in names:
            raise ValueError(f"unknown parameter {name!r}; expected one of {sorted(names)}")
        ranges[name] = (float(low), float(high))
    for name in scenario.parameter_names():
        ranges.setdefault(name, (scenario.get(name), scenario.get(name)))
    for name, (low, high) in ranges.items():
        if low > high:
            raise ValueError(f"range for {name} is inverted: {low} > {high}")
        if name == "capex" and low <= 0:
            raise ValueError("capex range must be strictly positive")
        if name == "risk_reduction" and not (0.0 <= low and high <= 1.0):
            raise ValueError("risk_reduction range must lie within [0, 1]")
        if name == "horizon_years" and low < 1:
            raise ValueError("horizon_years range must be at least 1")
        if name.startswith(COST_PREFIX) and low < 0:
            raise ValueError(f"cost range for {name} must be non-negative")
    return ranges


@dataclass(frozen=True)
class Interval:
    low: float
    high: float


@dataclass(frozen=True)
class Bounds:
    """Exact output bounds over an input box. ``inf`` payback = no benefit."""

    annual_cost: Interval
    annual_benefit: Interval
    payback_years: Interval
    roi: Interval
    net_roi: Interval


def _div(num: float, den: float) -> float:
    return math.inf if den == 0 else num / den


def interval_bounds(scenario: Scenario, ranges: Ranges) -> Bounds:
    """Tight bounds on every output over the box ``ranges``."""
    cost_lo = sum(ranges[COST_PREFIX + k][0] for k in scenario.model.components)
    cost_hi = sum(ranges[COST_PREFIX + k][1] for k in scenario.model.components)
    cap_lo, cap_hi = ranges["capex"]
    rr_lo, rr_hi = ranges["risk_reduction"]
    h_lo, h_hi = ranges["horizon_years"]
    h_lo, h_hi = int(round(h_lo)), int(round(h_hi))

    ben_lo, ben_hi = cost_lo * rr_lo, cost_hi * rr_hi
    roi_lo = ben_lo * h_lo / cap_hi
    roi_hi = ben_hi * h_hi / cap_lo
    return Bounds(
        annual_cost=Interval(cost_lo, cost_hi),
        annual_benefit=Interval(ben_lo, ben_hi),
        payback_years=Interval(_div(cap_lo, ben_hi), _div(cap_hi, ben_lo)),
        roi=Interval(roi_lo, roi_hi),
        net_roi=Interval(roi_lo - 1.0, roi_hi - 1.0),
    )


def hurdle_verdict(bounds: Bounds, hurdle_years: float) -> str:
    """Whether the payback hurdle holds everywhere, nowhere, or only in part of the box."""
    pb = bounds.payback_years
    if pb.high <= hurdle_years:
        return "passes for every input in the assumed ranges"
    if pb.low > hurdle_years:
        return "fails for every input in the assumed ranges"
    return "depends on where the inputs fall within the assumed ranges"


@dataclass(frozen=True)
class SimulationSummary:
    samples: int
    seed: int
    hurdle_years: float
    p_within_hurdle: float
    standard_error: float
    payback_p05: float
    payback_p50: float
    payback_p95: float
    net_roi_p05: float
    net_roi_p50: float
    net_roi_p95: float


def _nearest_rank(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 1]); well defined with inf values."""
    index = max(0, min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[index]


def simulate(
    scenario: Scenario,
    ranges: Ranges,
    hurdle_years: float = 3.0,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> SimulationSummary:
    """Seeded uniform sampling of the box; deterministic for a given seed."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = random.Random(seed)
    names = scenario.parameter_names()
    paybacks: List[float] = []
    net_rois: List[float] = []
    for _ in range(samples):
        draw = {n: rng.uniform(*ranges[n]) for n in names}
        cost = sum(draw[COST_PREFIX + k] for k in scenario.model.components)
        benefit = cost * draw["risk_reduction"]
        horizon = int(round(draw["horizon_years"]))
        paybacks.append(_div(draw["capex"], benefit))
        net_rois.append(benefit * horizon / draw["capex"] - 1.0)
    paybacks.sort()
    net_rois.sort()
    p = sum(1 for x in paybacks if x <= hurdle_years) / samples
    return SimulationSummary(
        samples=samples,
        seed=seed,
        hurdle_years=hurdle_years,
        p_within_hurdle=p,
        standard_error=math.sqrt(p * (1.0 - p) / samples),
        payback_p05=_nearest_rank(paybacks, 0.05),
        payback_p50=_nearest_rank(paybacks, 0.50),
        payback_p95=_nearest_rank(paybacks, 0.95),
        net_roi_p05=_nearest_rank(net_rois, 0.05),
        net_roi_p50=_nearest_rank(net_rois, 0.50),
        net_roi_p95=_nearest_rank(net_rois, 0.95),
    )


def corner_check(scenario: Scenario, ranges: Ranges) -> Tuple[float, float]:
    """Brute-force min and max payback over all 2^n box corners via evaluate().

    Used by the tests (and available to users) to confirm interval_bounds()
    against the model itself. Exponential in the number of varying inputs.
    """
    varying = [n for n in scenario.parameter_names() if ranges[n][0] != ranges[n][1]]
    best, worst = math.inf, -math.inf
    for mask in range(1 << len(varying)):
        s = scenario
        for i, name in enumerate(varying):
            s = s.with_value(name, ranges[name][(mask >> i) & 1])
        for name in scenario.parameter_names():
            if name not in varying:
                s = s.with_value(name, ranges[name][0])
        r = evaluate(s.model, s.capex, s.risk_reduction, s.horizon_years)
        pb = math.inf if r.payback_years is None else r.payback_years
        best, worst = min(best, pb), max(worst, pb)
    return best, worst
