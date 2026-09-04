"""
Monte Carlo propagation of input uncertainty through the capitals model.

The deterministic model takes point estimates and returns a point answer. That
answer carries an implied precision it has not earned: "payback in 1.59 years"
reads as though the 1.59 were measured. It was not. It is the arithmetic
consequence of seven cost estimates, a capital cost, and an assumed risk
reduction, none of which is known exactly.

This module replaces each point input with a distribution, draws many samples,
and reports the distribution of the answer -- in particular, the probability
that the investment pays back inside a stated number of years.

WHAT THIS DOES NOT DO
---------------------
Propagating uncertainty is not the same as reducing it. Nothing here makes the
risk-reduction assumption more credible. If you feed the model a distribution
centred on the wrong number, you get a confident distribution centred on the
wrong answer, and the spread will make it look rigorous. The output is a
statement about your stated beliefs, not about the world.

Two structural assumptions are worth stating plainly:

  Independence. Inputs are sampled independently. In reality the cost
  components are driven by a shared underlying incident rate, so a bad year is
  bad across several lines at once. Independent sampling lets those errors
  cancel, which understates the spread of total annual cost. Treat the reported
  intervals as narrower than the truth.

  Stationarity. One risk reduction and one annual cost are assumed to hold
  unchanged across the whole horizon. Controls degrade, production rates move,
  and workforces turn over.

Determinism: every function takes an explicit seed defaulting to
:data:`DEFAULT_SEED`, and draws come from ``numpy.random.default_rng``. The
same seed and sample count reproduce the same numbers exactly.
"""

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from capitals import CostModel, evaluate

#: Default RNG seed. 1904 after 29 CFR 1904, the OSHA recordkeeping rule.
#: The value carries no meaning beyond making runs reproducible.
DEFAULT_SEED = 1904

#: Default number of Monte Carlo draws.
DEFAULT_SAMPLES = 50_000


# --------------------------------------------------------------------------
# Distributions
# --------------------------------------------------------------------------


class Distribution:
    """Base class for an input distribution.

    Subclasses implement :meth:`sample`. Instances are immutable and carry no
    RNG state; the generator is passed in, so reproducibility is controlled at
    one place by the caller.
    """

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw ``size`` values."""
        raise NotImplementedError

    @property
    def mean_estimate(self) -> float:
        """A representative central value, used for baseline comparison."""
        raise NotImplementedError


@dataclass(frozen=True)
class Fixed(Distribution):
    """A value known exactly, or one deliberately held constant."""

    value: float

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return np.full(size, float(self.value))

    @property
    def mean_estimate(self) -> float:
        return float(self.value)


@dataclass(frozen=True)
class Uniform(Distribution):
    """Equal probability across ``[low, high]``.

    Appropriate when a range is defensible but no value inside it is more
    plausible than another. That is rarer than its popularity suggests.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError("high must not be less than low")

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.uniform(self.low, self.high, size)

    @property
    def mean_estimate(self) -> float:
        return (self.low + self.high) / 2.0


@dataclass(frozen=True)
class Triangular(Distribution):
    """Minimum, most likely, maximum -- the standard elicitation shape.

    Useful because the three numbers are ones a plant manager can actually
    answer. Its sharp peak and hard bounds are not physically meaningful.
    """

    low: float
    mode: float
    high: float

    def __post_init__(self) -> None:
        if not self.low <= self.mode <= self.high:
            raise ValueError("require low <= mode <= high")

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if self.low == self.high:
            return np.full(size, float(self.low))
        return rng.triangular(self.low, self.mode, self.high, size)

    @property
    def mean_estimate(self) -> float:
        return (self.low + self.mode + self.high) / 3.0


@dataclass(frozen=True)
class PERT(Distribution):
    """Beta-PERT: a smoother alternative to :class:`Triangular`.

    Same three elicited numbers, but weights the mode more heavily and rounds
    the shoulders, which is generally a better description of an expert's
    belief than a triangle. ``lam`` controls concentration; 4 is conventional.
    """

    low: float
    mode: float
    high: float
    lam: float = 4.0

    def __post_init__(self) -> None:
        if not self.low <= self.mode <= self.high:
            raise ValueError("require low <= mode <= high")
        if self.lam <= 0:
            raise ValueError("lam must be positive")

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        span = self.high - self.low
        if span == 0:
            return np.full(size, float(self.low))
        alpha = 1.0 + self.lam * (self.mode - self.low) / span
        beta = 1.0 + self.lam * (self.high - self.mode) / span
        return self.low + span * rng.beta(alpha, beta, size)

    @property
    def mean_estimate(self) -> float:
        return (self.low + self.lam * self.mode + self.high) / (self.lam + 2.0)


@dataclass(frozen=True)
class Normal(Distribution):
    """Gaussian, truncated by resampling to stay inside ``[lower, upper]``.

    Truncation is by rejection and resampling rather than clipping, so no
    probability mass piles up on the bounds. Bounds default to non-negative
    and unbounded above, which is what cost figures need.
    """

    mean: float
    sd: float
    lower: float = 0.0
    upper: float = math.inf

    def __post_init__(self) -> None:
        if self.sd < 0:
            raise ValueError("sd must be non-negative")
        if self.lower > self.upper:
            raise ValueError("lower must not exceed upper")

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if self.sd == 0:
            return np.full(size, float(np.clip(self.mean, self.lower, self.upper)))
        out = rng.normal(self.mean, self.sd, size)
        # Resample out-of-bounds draws. Capped so a badly specified
        # distribution fails loudly rather than hanging.
        for _ in range(100):
            bad = (out < self.lower) | (out > self.upper)
            n_bad = int(bad.sum())
            if n_bad == 0:
                return out
            out[bad] = rng.normal(self.mean, self.sd, n_bad)
        raise ValueError(
            "truncated normal failed to converge: the bounds exclude almost "
            "all of the distribution's mass"
        )

    @property
    def mean_estimate(self) -> float:
        return float(np.clip(self.mean, self.lower, self.upper))


@dataclass(frozen=True)
class BoundedBeta(Distribution):
    """A Beta distribution on ``[0, 1]``, for the risk-reduction input.

    Parameterised by mean and a concentration, which are easier to reason
    about than raw alpha/beta. Higher concentration is a tighter belief.
    """

    mean: float
    concentration: float = 20.0

    def __post_init__(self) -> None:
        if not 0.0 < self.mean < 1.0:
            raise ValueError("mean must be strictly between 0 and 1")
        if self.concentration <= 0:
            raise ValueError("concentration must be positive")

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        alpha = self.mean * self.concentration
        beta = (1.0 - self.mean) * self.concentration
        return rng.beta(alpha, beta, size)

    @property
    def mean_estimate(self) -> float:
        return float(self.mean)


# --------------------------------------------------------------------------
# Specification and results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MonteCarloSpec:
    """A distribution for every model input."""

    label: str
    components: Dict[str, Distribution]
    capex: Distribution
    risk_reduction: Distribution
    horizon_years: int = 5

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("at least one cost component is required")
        if self.horizon_years <= 0:
            raise ValueError("horizon_years must be positive")

    def baseline_model(self) -> CostModel:
        """The point-estimate model implied by the distribution centres."""
        return CostModel(
            self.label,
            {k: d.mean_estimate for k, d in self.components.items()},
        )

    def baseline_result(self):
        """Deterministic result at the distribution centres.

        Useful as a reference point, but note it is not the mean of the Monte
        Carlo output: payback is a ratio, so its expectation does not equal the
        ratio of expectations (Jensen's inequality). Expect the simulated mean
        payback to sit above this value.
        """
        return evaluate(
            self.baseline_model(),
            self.capex.mean_estimate,
            self.risk_reduction.mean_estimate,
            self.horizon_years,
        )


@dataclass(frozen=True)
class MonteCarloResult:
    """Simulated outcomes. Arrays are aligned element-wise by draw."""

    label: str
    samples: int
    seed: int
    horizon_years: int
    annual_cost: np.ndarray
    capex: np.ndarray
    risk_reduction: np.ndarray
    annual_benefit: np.ndarray
    payback_years: np.ndarray  # +inf where the benefit is zero
    roi: np.ndarray
    net_roi: np.ndarray

    def probability_payback_within(self, years: float) -> float:
        """Share of draws that recover the capital cost within ``years``.

        Raises:
            ValueError: if ``years`` is not positive.
        """
        if years <= 0:
            raise ValueError("years must be positive")
        return float(np.mean(self.payback_years <= years))

    def probability_roi_at_least(self, threshold: float) -> float:
        """Share of draws whose cumulative-benefit ROI reaches ``threshold``.

        ``threshold`` is a ratio: 1.0 means benefits equal the investment.
        """
        return float(np.mean(self.roi >= threshold))

    def probability_never_pays_back(self) -> float:
        """Share of draws with zero annual benefit and so no payback."""
        return float(np.mean(np.isinf(self.payback_years)))

    def percentiles(
        self, metric: str = "payback_years", qs: Sequence[float] = (5, 25, 50, 75, 95)
    ) -> Dict[float, float]:
        """Percentiles of one output array.

        Percentiles are computed on finite draws only. If some draws never pay
        back, the finite percentiles understate risk, so check
        :meth:`probability_never_pays_back` alongside them.

        Raises:
            ValueError: on an unknown metric or if no draws are finite.
        """
        if metric not in ("payback_years", "roi", "net_roi", "annual_benefit",
                          "annual_cost", "capex", "risk_reduction"):
            raise ValueError(f"unknown metric {metric!r}")
        values = getattr(self, metric)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"no finite values for {metric!r}")
        return {q: float(np.percentile(finite, q)) for q in qs}

    def summary(self, hurdle_years: float = 3.0) -> Dict[str, float]:
        """A compact dictionary of the headline figures."""
        pct = self.percentiles("payback_years")
        return {
            "samples": float(self.samples),
            "seed": float(self.seed),
            "p_payback_within_hurdle": self.probability_payback_within(hurdle_years),
            "hurdle_years": float(hurdle_years),
            "p_never_pays_back": self.probability_never_pays_back(),
            "payback_p5": pct[5],
            "payback_median": pct[50],
            "payback_p95": pct[95],
            "median_roi": float(np.median(self.roi)),
            "median_annual_cost": float(np.median(self.annual_cost)),
        }


def run(
    spec: MonteCarloSpec,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> MonteCarloResult:
    """Draw samples and propagate them through the model.

    The arithmetic is vectorised rather than looping through
    :func:`capitals.evaluate`, so the formulas are duplicated here. The test
    suite pins the two implementations against each other to keep them honest.

    Args:
        spec: distributions for every input.
        samples: number of draws.
        seed: RNG seed. Fixed by default so results are reproducible.

    Raises:
        ValueError: if ``samples`` is not positive, or if any draw is invalid
            (negative cost, capex at or below zero, risk reduction outside
            ``[0, 1]``) -- which indicates a badly specified distribution.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")

    rng = np.random.default_rng(seed)

    # Sample in a fixed, sorted order so that adding an unrelated component
    # does not silently change the draws for existing ones.
    component_draws = {
        name: spec.components[name].sample(rng, samples)
        for name in sorted(spec.components)
    }
    annual_cost = np.sum(np.stack(list(component_draws.values())), axis=0)
    capex = spec.capex.sample(rng, samples)
    risk_reduction = spec.risk_reduction.sample(rng, samples)

    if np.any(annual_cost < 0):
        raise ValueError("sampled a negative annual cost; check component bounds")
    if np.any(capex <= 0):
        raise ValueError("sampled a non-positive capex; check the capex distribution")
    if np.any((risk_reduction < 0) | (risk_reduction > 1)):
        raise ValueError(
            "sampled a risk_reduction outside [0, 1]; use BoundedBeta or bound "
            "the distribution"
        )

    annual_benefit = annual_cost * risk_reduction
    total_benefit = annual_benefit * spec.horizon_years

    payback = np.full(samples, np.inf)
    positive = annual_benefit > 0
    payback[positive] = capex[positive] / annual_benefit[positive]

    roi = total_benefit / capex
    net_roi = (total_benefit - capex) / capex

    return MonteCarloResult(
        label=spec.label,
        samples=samples,
        seed=seed,
        horizon_years=spec.horizon_years,
        annual_cost=annual_cost,
        capex=capex,
        risk_reduction=risk_reduction,
        annual_benefit=annual_benefit,
        payback_years=payback,
        roi=roi,
        net_roi=net_roi,
    )


# --------------------------------------------------------------------------
# Illustrative specification
# --------------------------------------------------------------------------


def illustrative_spec(
    model: CostModel,
    capex: float,
    risk_reduction: float,
    horizon_years: int = 5,
    cost_spread: float = 0.30,
    capex_spread: float = 0.15,
    risk_concentration: float = 20.0,
) -> MonteCarloSpec:
    """Wrap point estimates in uncertainty bands, for demonstration only.

    THE SPREADS BELOW ARE NOT MEASURED. They are placeholders that make the
    machinery runnable and show the shape of the output. They are not derived
    from any dataset, any published study, or any elicitation. A real analysis
    replaces every one of them with a range someone is prepared to defend.

    The defaults encode three ordinary-looking guesses: cost lines are
    uncertain by roughly a third either way, a quoted capital cost is firmer
    than an estimated loss, and capital projects overrun more often than they
    underrun -- so capex is skewed with its mode at the quote and a longer
    upside tail.

    Args:
        model: point estimates for each cost component.
        capex: point estimate of the capital cost.
        risk_reduction: assumed proportional reduction, in ``(0, 1)``.
        horizon_years: evaluation period.
        cost_spread: half-width of the PERT band on each cost line.
        capex_spread: downside half-width on capex; the upside is twice this.
        risk_concentration: Beta concentration on risk reduction. Higher is a
            tighter belief; 20 gives roughly +/-0.09 either side of 0.80.

    Raises:
        ValueError: if the spreads are negative or the risk reduction is not
            strictly inside ``(0, 1)``.
    """
    if cost_spread < 0 or capex_spread < 0:
        raise ValueError("spreads must be non-negative")
    if not 0.0 < risk_reduction < 1.0:
        raise ValueError(
            "risk_reduction must be strictly between 0 and 1 for a Beta prior"
        )

    components: Dict[str, Distribution] = {
        name: PERT(
            low=value * (1.0 - cost_spread),
            mode=value,
            high=value * (1.0 + cost_spread),
        )
        for name, value in model.components.items()
    }
    return MonteCarloSpec(
        label=model.label,
        components=components,
        capex=PERT(
            low=capex * (1.0 - capex_spread),
            mode=capex,
            high=capex * (1.0 + 2.0 * capex_spread),
        ),
        risk_reduction=BoundedBeta(
            mean=risk_reduction, concentration=risk_concentration
        ),
        horizon_years=horizon_years,
    )


def format_result(result: MonteCarloResult, hurdle_years: float = 3.0) -> str:
    """Render a Monte Carlo result as fixed-width text."""
    pct = result.percentiles("payback_years")
    lines = [
        f"{result.label}  ({result.samples:,} draws, seed {result.seed})",
        f"  P(payback within {hurdle_years:g} yr)   "
        f"{result.probability_payback_within(hurdle_years):>7.1%}",
        f"  P(never pays back)         "
        f"{result.probability_never_pays_back():>7.1%}",
        "  payback years   "
        f"p5 {pct[5]:.2f}   p50 {pct[50]:.2f}   p95 {pct[95]:.2f}",
        f"  {result.horizon_years}-yr ROI       "
        f"p5 {np.percentile(result.roi, 5):.0%}   "
        f"p50 {np.percentile(result.roi, 50):.0%}   "
        f"p95 {np.percentile(result.roi, 95):.0%}",
    ]
    return "\n".join(lines)
