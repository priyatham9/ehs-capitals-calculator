"""
Deterministic sensitivity analysis for the capitals cost-benefit model.

Answers one question: which input, moved across a plausible range, most changes
the answer? And the sharper follow-up: which inputs can move the answer across
a decision threshold at all?

Two tools:

  tornado()      one-at-a-time perturbation. Each input is swung low-to-high
                 while every other input is held at baseline, and the resulting
                 swing in the output metric is recorded. Sorting by swing gives
                 the familiar tornado ordering.

  flip_point()   bisection for the exact input value at which a stated
                 conclusion (e.g. "payback within 3 years") changes sign.

One-at-a-time analysis is the right tool for a model this small and this
transparent, but it is worth being clear about what it cannot do. It holds all
other inputs fixed, so it does not explore interactions, and it will understate
joint risk whenever inputs move together -- which, for cost components driven
by the same underlying incident rate, they certainly do. Use montecarlo.py when
inputs should move at once.

The ranking is also only as meaningful as the ranges fed to it. A tornado chart
built from arbitrary +/-30% bands ranks inputs by the width of a band someone
chose, not by real-world uncertainty. Supply ranges you can defend.
"""

import math
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from capitals import CostModel, Result, evaluate

COST_PREFIX = "cost:"

#: Metrics that can be used as the tornado output.
METRICS = ("payback_years", "roi", "net_roi")

#: Metrics for which a *smaller* value is a better outcome.
_LOWER_IS_BETTER = frozenset({"payback_years"})


# --------------------------------------------------------------------------
# Scenario: a fully specified set of model inputs that can be perturbed.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One complete set of inputs to :func:`capitals.evaluate`.

    Inputs are addressed by string name so that a single perturbation routine
    can walk every input uniformly. Cost components are namespaced with a
    ``cost:`` prefix, e.g. ``"cost:lost_work_time"``.
    """

    model: CostModel
    capex: float
    risk_reduction: float
    horizon_years: int = 5

    # -- introspection -----------------------------------------------------

    def parameter_names(self) -> List[str]:
        """Every perturbable input, cost components first."""
        return [COST_PREFIX + k for k in self.model.components] + [
            "capex",
            "risk_reduction",
            "horizon_years",
        ]

    def get(self, name: str) -> float:
        """Current value of one input.

        Raises:
            KeyError: if ``name`` is not an input of this scenario.
        """
        if name.startswith(COST_PREFIX):
            key = name[len(COST_PREFIX) :]
            if key not in self.model.components:
                raise KeyError(f"no cost component named {key!r}")
            return float(self.model.components[key])
        if name in ("capex", "risk_reduction", "horizon_years"):
            return float(getattr(self, name))
        raise KeyError(f"unknown parameter {name!r}")

    def is_integer_parameter(self, name: str) -> bool:
        """True for inputs that only take whole-number values."""
        return name == "horizon_years"

    # -- perturbation ------------------------------------------------------

    def with_value(self, name: str, value: float) -> "Scenario":
        """Return a copy of this scenario with one input changed."""
        if name.startswith(COST_PREFIX):
            key = name[len(COST_PREFIX) :]
            if key not in self.model.components:
                raise KeyError(f"no cost component named {key!r}")
            components = dict(self.model.components)
            components[key] = float(value)
            return replace(
                self, model=CostModel(self.model.label, components)
            )
        if name == "capex":
            return replace(self, capex=float(value))
        if name == "risk_reduction":
            return replace(self, risk_reduction=float(value))
        if name == "horizon_years":
            return replace(self, horizon_years=int(round(value)))
        raise KeyError(f"unknown parameter {name!r}")

    def evaluate(self) -> Result:
        """Run the underlying cost-benefit model on these inputs."""
        return evaluate(
            self.model, self.capex, self.risk_reduction, self.horizon_years
        )


def metric_value(result: Result, metric: str = "payback_years") -> float:
    """Extract one output metric as a plain float.

    ``None`` results -- which the model returns when the annual benefit is
    zero and the investment therefore never pays back -- are mapped to
    ``+inf`` for payback and to ``0.0`` for the two ROI measures, so that
    downstream arithmetic and sorting stay well defined.

    Raises:
        ValueError: on an unknown metric name.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    raw = getattr(result, metric)
    if raw is not None:
        return float(raw)
    return math.inf if metric == "payback_years" else 0.0


# --------------------------------------------------------------------------
# Ranges
# --------------------------------------------------------------------------


def relative_ranges(
    scenario: Scenario,
    fraction: float = 0.30,
    include_horizon: bool = True,
) -> Dict[str, Tuple[float, float]]:
    """Build symmetric +/- ranges around each input's baseline.

    This is a convenience for exploration, not a substitute for judgement.
    Every input gets the same relative band, which is almost never true of
    real uncertainty. ``risk_reduction`` is clamped to ``[0, 1]`` and costs and
    capex are floored at zero.

    Args:
        scenario: the baseline.
        fraction: half-width of the band, as a proportion. 0.30 means +/-30%.
        include_horizon: whether to vary the evaluation horizon. Horizon has no
            effect on payback, so it appears as a zero-swing bar there.

    Raises:
        ValueError: if ``fraction`` is negative.
    """
    if fraction < 0:
        raise ValueError("fraction must be non-negative")

    ranges: Dict[str, Tuple[float, float]] = {}
    for name in scenario.parameter_names():
        if name == "horizon_years" and not include_horizon:
            continue
        base = scenario.get(name)
        low = base * (1.0 - fraction)
        high = base * (1.0 + fraction)
        if name == "risk_reduction":
            low, high = max(0.0, low), min(1.0, high)
        elif name == "horizon_years":
            low, high = max(1.0, round(low)), max(1.0, round(high))
        else:
            low, high = max(0.0, low), max(0.0, high)
        ranges[name] = (low, high)
    return ranges


# --------------------------------------------------------------------------
# Tornado
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TornadoBar:
    """The effect of swinging one input across its range."""

    parameter: str
    baseline_input: float
    low_input: float
    high_input: float
    baseline_output: float
    low_output: float
    high_output: float
    metric: str

    @property
    def swing(self) -> float:
        """Absolute distance between the two endpoint outputs.

        ``inf`` if either endpoint never pays back, which is itself the most
        important thing a sensitivity run can tell you.
        """
        if math.isinf(self.low_output) or math.isinf(self.high_output):
            return math.inf
        return abs(self.high_output - self.low_output)

    @property
    def worst_output(self) -> float:
        """The endpoint output that represents the less favourable outcome."""
        if self.metric in _LOWER_IS_BETTER:
            return max(self.low_output, self.high_output)
        return min(self.low_output, self.high_output)

    @property
    def best_output(self) -> float:
        """The endpoint output that represents the more favourable outcome."""
        if self.metric in _LOWER_IS_BETTER:
            return min(self.low_output, self.high_output)
        return max(self.low_output, self.high_output)


def tornado(
    scenario: Scenario,
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    metric: str = "payback_years",
) -> List[TornadoBar]:
    """Rank inputs by how much they move the output metric.

    Args:
        scenario: baseline inputs.
        ranges: per-input ``(low, high)`` bounds. Defaults to +/-30% via
            :func:`relative_ranges`.
        metric: one of :data:`METRICS`.

    Returns:
        Bars sorted by swing, widest first. Ties keep scenario input order,
        so the result is deterministic.

    Raises:
        ValueError: on an unknown metric.
        KeyError: if ``ranges`` names an input the scenario does not have.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    if ranges is None:
        ranges = relative_ranges(scenario)

    baseline_output = metric_value(scenario.evaluate(), metric)

    bars: List[TornadoBar] = []
    for name, (low, high) in ranges.items():
        scenario.get(name)  # raises KeyError on an unknown input
        bars.append(
            TornadoBar(
                parameter=name,
                baseline_input=scenario.get(name),
                low_input=low,
                high_input=high,
                baseline_output=baseline_output,
                low_output=metric_value(
                    scenario.with_value(name, low).evaluate(), metric
                ),
                high_output=metric_value(
                    scenario.with_value(name, high).evaluate(), metric
                ),
                metric=metric,
            )
        )

    # Sort by swing descending, stably, so equal swings keep insertion order.
    order = {name: i for i, name in enumerate(ranges)}
    bars.sort(key=lambda b: (-_sortable(b.swing), order[b.parameter]))
    return bars


def _sortable(x: float) -> float:
    """Map +inf to a large finite value so sorting is total."""
    return 1e308 if math.isinf(x) else x


# --------------------------------------------------------------------------
# Decision flipping
# --------------------------------------------------------------------------


def flip_point(
    scenario: Scenario,
    parameter: str,
    predicate: Callable[[Result], bool],
    low: float,
    high: float,
    tolerance: float = 1e-9,
    max_iterations: int = 200,
) -> Optional[float]:
    """Find the input value at which a conclusion changes.

    Bisects ``[low, high]`` for the boundary where ``predicate`` flips. For
    integer-valued inputs the range is scanned exhaustively instead, since
    bisection on a step function returns a meaningless fractional answer.

    Args:
        scenario: baseline inputs.
        parameter: which input to vary.
        predicate: the conclusion, as a function of the model result. For
            example ``lambda r: r.payback_years is not None and
            r.payback_years <= 3``.
        low: lower bound of the search.
        high: upper bound of the search.
        tolerance: absolute width at which bisection stops.
        max_iterations: bisection cap.

    Returns:
        The boundary value, or ``None`` if the conclusion holds at both ends
        (or fails at both ends) and therefore never flips inside the range.

    Raises:
        ValueError: if ``low`` exceeds ``high``.
        KeyError: on an unknown parameter.
    """
    if low > high:
        raise ValueError("low must not exceed high")

    def holds(value: float) -> bool:
        return predicate(scenario.with_value(parameter, value).evaluate())

    low_holds, high_holds = holds(low), holds(high)
    if low_holds == high_holds:
        return None

    if scenario.is_integer_parameter(parameter):
        # Step function: walk integers and return the first value whose
        # conclusion differs from the low end.
        for candidate in range(int(math.ceil(low)), int(math.floor(high)) + 1):
            if holds(float(candidate)) != low_holds:
                return float(candidate)
        return None

    for _ in range(max_iterations):
        if high - low <= tolerance:
            break
        mid = (low + high) / 2.0
        if holds(mid) == low_holds:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def payback_within(years: float) -> Callable[[Result], bool]:
    """Predicate factory: does the investment pay back within ``years``?

    Raises:
        ValueError: if ``years`` is not positive.
    """
    if years <= 0:
        raise ValueError("years must be positive")

    def predicate(result: Result) -> bool:
        return result.payback_years is not None and result.payback_years <= years

    return predicate


@dataclass(frozen=True)
class FlipReport:
    """Whether one input can, on its own, overturn a decision."""

    parameter: str
    baseline_input: float
    baseline_holds: bool
    low_input: float
    high_input: float
    flip_input: Optional[float]

    @property
    def can_flip(self) -> bool:
        """True if the conclusion changes somewhere inside the search range."""
        return self.flip_input is not None


def decision_sensitivity(
    scenario: Scenario,
    predicate: Callable[[Result], bool],
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[FlipReport]:
    """For each input, report whether moving it alone overturns the decision.

    This is usually more decision-relevant than a tornado chart. A wide swing
    in payback does not matter if every value in the range clears the hurdle;
    a narrow swing matters enormously if it straddles it.

    Args:
        scenario: baseline inputs.
        predicate: the conclusion under test.
        ranges: per-input search bounds. Defaults to +/-30%.

    Returns:
        One report per input, flip-capable inputs first.
    """
    if ranges is None:
        ranges = relative_ranges(scenario)

    baseline_holds = predicate(scenario.evaluate())
    reports: List[FlipReport] = []
    for name, (low, high) in ranges.items():
        reports.append(
            FlipReport(
                parameter=name,
                baseline_input=scenario.get(name),
                baseline_holds=baseline_holds,
                low_input=low,
                high_input=high,
                flip_input=flip_point(scenario, name, predicate, low, high),
            )
        )
    order = {name: i for i, name in enumerate(ranges)}
    reports.sort(key=lambda r: (not r.can_flip, order[r.parameter]))
    return reports


# --------------------------------------------------------------------------
# Two-way grid
# --------------------------------------------------------------------------


def grid(
    scenario: Scenario,
    x_parameter: str,
    x_values: Sequence[float],
    y_parameter: str,
    y_values: Sequence[float],
    metric: str = "payback_years",
) -> List[List[float]]:
    """Evaluate the metric over a two-input grid.

    Returns a row-major table indexed ``[y][x]`` -- rows follow ``y_values``,
    columns follow ``x_values`` -- which is the orientation a rendered table
    or heat map wants.

    Raises:
        ValueError: on an unknown metric.
        KeyError: on an unknown parameter.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    scenario.get(x_parameter)
    scenario.get(y_parameter)

    table: List[List[float]] = []
    for y in y_values:
        row = [
            metric_value(
                scenario.with_value(y_parameter, y)
                .with_value(x_parameter, x)
                .evaluate(),
                metric,
            )
            for x in x_values
        ]
        table.append(row)
    return table


# --------------------------------------------------------------------------
# Text rendering
# --------------------------------------------------------------------------


def format_tornado(bars: Sequence[TornadoBar], width: int = 44) -> str:
    """Render tornado bars as fixed-width text for terminal output."""
    if not bars:
        return "(no parameters varied)"

    finite = [b.swing for b in bars if not math.isinf(b.swing)]
    largest = max(finite) if finite else 1.0
    if largest <= 0:
        largest = 1.0

    label_width = max(len(b.parameter) for b in bars)
    metric = bars[0].metric
    unit = "yr" if metric == "payback_years" else ""

    lines = [f"Swing in {metric} (one input at a time, others held at baseline)"]
    for bar in bars:
        if math.isinf(bar.swing):
            filled, swing_text = width, "never pays back at one end"
        else:
            filled = int(round(width * bar.swing / largest))
            swing_text = (
                f"{bar.swing:,.2f} {unit}".strip()
                if metric == "payback_years"
                else f"{bar.swing:,.1%}"
            )
        glyphs = "#" * filled + "." * (width - filled)
        lines.append(f"  {bar.parameter:<{label_width}}  {glyphs}  {swing_text}")
    return "\n".join(lines)
