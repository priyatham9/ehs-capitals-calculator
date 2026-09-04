"""
Human-capital-inclusive cost-benefit model for EHS investment.

Implements the loading-station comparison published in Brandon, C. (27 May
2025), "You Can't Manage What You Don't Value: Reimagining Capital in
Organizations", LeadingEHS.com. See SOURCE_URL below. The article's cost
table, its line items, and its published payback and ROI figures were
retrieved and checked against this implementation on 3 September 2026.

The model contrasts two accountings of the same safety investment:

  Traditional - counts only directly booked costs (equipment downtime,
                  material loss, injury costs).
  Human-capital - additionally counts costs that are real but rarely booked
                  against the incident (lost work time, turnover-driven
                  retraining, productivity loss, morale effects).

The arithmetic is deliberately simple and fully transparent:

    annual_benefit = annual_incident_cost * risk_reduction
    residual_cost  = annual_incident_cost - annual_benefit
    payback_years  = capex / annual_benefit
    roi            = (annual_benefit * horizon_years) / capex

ROI here is cumulative-benefit-to-investment, not net return. That is the
convention used in the source article; `net_roi` is also reported for readers
who expect the net definition. Neither figure is discounted: a dollar saved in
year 5 is counted the same as a dollar saved in year 1. See the README for what
that costs you.

`risk_reduction` is an assumption the user supplies. Nothing in this module
estimates it, validates it, or knows whether it is achievable.
"""

from dataclasses import dataclass
from typing import Dict, Optional

SOURCE_URL = (
    "https://leadingehs.com/2025/05/27/"
    "you-cant-manage-what-you-dont-value-reimagining-capital-in-organizations/"
)


@dataclass(frozen=True)
class CostModel:
    """A named bundle of annual incident cost components."""

    label: str
    components: Dict[str, float]

    @property
    def annual_cost(self) -> float:
        return float(sum(self.components.values()))


@dataclass(frozen=True)
class Result:
    label: str
    annual_cost: float
    annual_benefit: float
    payback_years: Optional[float]
    roi: Optional[float]
    net_roi: Optional[float]
    horizon_years: int
    capex: float

    @property
    def residual_cost(self) -> float:
        """Annual incident cost still expected after the control is in place."""
        return self.annual_cost - self.annual_benefit

    @property
    def total_benefit(self) -> float:
        """Undiscounted benefit accumulated over the horizon."""
        return self.annual_benefit * self.horizon_years

    def __str__(self) -> str:
        pb = "never" if self.payback_years is None else f"{self.payback_years:.2f} yr"
        roi = "n/a" if self.roi is None else f"{self.roi:.0%}"
        return (
            f"{self.label:<23} cost/yr ${self.annual_cost:>10,.0f}   "
            f"benefit/yr ${self.annual_benefit:>10,.0f}   "
            f"payback {pb:>9}   {self.horizon_years}-yr ROI {roi:>7}"
        )


def evaluate(
    model: CostModel,
    capex: float,
    risk_reduction: float,
    horizon_years: int = 5,
) -> Result:
    """Evaluate one cost accounting against a capital investment.

    Args:
        model: the annual incident cost components to count.
        capex: up-front capital investment.
        risk_reduction: expected proportional reduction in incident cost,
            in [0, 1]. 0.8 means an 80% reduction, i.e. 20% residual risk.
        horizon_years: evaluation period for ROI.

    Raises:
        ValueError: on inputs that would make the result meaningless.
    """
    if capex <= 0:
        raise ValueError("capex must be positive")
    if not 0.0 <= risk_reduction <= 1.0:
        raise ValueError("risk_reduction must be a proportion in [0, 1]")
    if horizon_years <= 0:
        raise ValueError("horizon_years must be positive")
    if any(v < 0 for v in model.components.values()):
        raise ValueError("cost components must be non-negative")

    annual_cost = model.annual_cost
    annual_benefit = annual_cost * risk_reduction

    if annual_benefit == 0:
        return Result(
            label=model.label,
            annual_cost=annual_cost,
            annual_benefit=0.0,
            payback_years=None,
            roi=None,
            net_roi=None,
            horizon_years=horizon_years,
            capex=capex,
        )

    total_benefit = annual_benefit * horizon_years
    return Result(
        label=model.label,
        annual_cost=annual_cost,
        annual_benefit=annual_benefit,
        payback_years=capex / annual_benefit,
        roi=total_benefit / capex,
        net_roi=(total_benefit - capex) / capex,
        horizon_years=horizon_years,
        capex=capex,
    )


# --- The worked example from the source article -----------------------------
# A loading-station safety investment. Every line item below is transcribed
# from the cost table published in the article; none of it is invented or
# back-solved. The article's own table also reports the intermediate rows
# (residual cost at 20%, annual cost savings), which are pinned in the tests.

TRADITIONAL = CostModel(
    label="Traditional",
    components={
        "equipment_downtime": 30_000.0,
        "material_loss_spills": 10_000.0,
        "injury_costs": 15_000.0,
    },
)

HUMAN_CAPITAL = CostModel(
    label="Human-capital-inclusive",
    components={
        **TRADITIONAL.components,
        "lost_work_time": 20_000.0,
        "retraining_turnover": 8_000.0,
        "productivity_loss": 25_000.0,
        "morale_team_performance": 10_000.0,
    },
)

LOADING_STATION_CAPEX = 150_000.0
LOADING_STATION_RISK_REDUCTION = 0.80


def worked_example(horizon_years: int = 5):
    """Reproduce the published loading-station comparison."""
    return [
        evaluate(m, LOADING_STATION_CAPEX, LOADING_STATION_RISK_REDUCTION, horizon_years)
        for m in (TRADITIONAL, HUMAN_CAPITAL)
    ]


if __name__ == "__main__":
    print("Loading-station safety investment")
    print(f"  capital cost      ${LOADING_STATION_CAPEX:,.0f}")
    print(f"  risk reduction    {LOADING_STATION_RISK_REDUCTION:.0%}")
    print()
    for r in worked_example():
        print(" ", r)
