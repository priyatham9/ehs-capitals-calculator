"""
Heat-illness exposure calculator: NWS heat index plus OSHA's proposed triggers.

Standard library only. Two parts:

  heat_index()     the National Weather Service heat index (deg F) from air
                   temperature (deg F) and relative humidity (%), using the
                   NWS procedure: the Steadman simple formula, then the
                   Rothfusz regression with the two NWS adjustments when the
                   simple result averaged with the temperature is >= 80 F.

  classify()       compares a heat index with the initial and high heat
                   triggers in OSHA's proposed rule.

Sources
-------
NWS regression and adjustments: https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
  (the regression is also published at https://www.weather.gov/ama/heatindex,
  which states an error of +/-1.3 F).

OSHA proposed rule: "Heat Injury and Illness Prevention in Outdoor and Indoor
Work Settings", 89 FR 70698 (30 August 2024), Docket OSHA-2021-0009. Proposed
29 CFR 1910.148(b) defines:
  initial heat trigger = heat index of 80 F, or a wet bulb globe temperature
                         (WBGT) equal to the NIOSH Recommended Alert Limit
  high heat trigger    = heat index of 90 F, or a WBGT equal to the NIOSH
                         Recommended Exposure Limit
Proposed (f)(2): at the high heat trigger, a minimum 15-minute paid rest break
at least every two hours. Proposed (f)(5): warning signs at indoor work areas
with ambient temperatures that regularly exceed 120 F.
Regulatory text: https://www.osha.gov/sites/default/files/Heat-NPRM-Final-Reg-Text.pdf

Limits you should know about
----------------------------
* This is a proposal. It is not in force; the final text may differ.
* Only the heat-index trigger is implemented. The WBGT triggers depend on NIOSH
  limits that vary with work rate and acclimatization, and no numeric WBGT
  value is hard-coded here because none could be verified in the cited text.
* NWS heat index values assume shade and light wind; direct sun can raise the
  apparent value by up to 15 F (NWS). Sun is not modelled.
* The NWS regression is not valid for extreme combinations of temperature and
  humidity outside the data it was fitted to; inputs are range-checked only
  for plausibility (0-100 % RH).
"""

import math
from typing import Dict

#: Proposed 29 CFR 1910.148(b): heat index (deg F) for the initial heat trigger.
INITIAL_HEAT_TRIGGER_F = 80.0
#: Proposed 29 CFR 1910.148(b): heat index (deg F) for the high heat trigger.
HIGH_HEAT_TRIGGER_F = 90.0
#: Proposed 29 CFR 1910.148(f)(5): ambient temperature for indoor warning signs.
INDOOR_WARNING_SIGN_F = 120.0

OSHA_SOURCE = (
    "OSHA proposed rule, Heat Injury and Illness Prevention in Outdoor and "
    "Indoor Work Settings, 89 FR 70698 (30 Aug 2024), Docket OSHA-2021-0009, "
    "proposed 29 CFR 1910.148(b)"
)
NWS_SOURCE = "https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml"

NONE, INITIAL, HIGH = "below_initial", "initial", "high"


def heat_index(temp_f: float, rh: float) -> float:
    """NWS heat index in deg F for air temperature ``temp_f`` and humidity ``rh`` (%).

    Raises:
        ValueError: if either input is non-finite or ``rh`` is outside 0-100.
    """
    if not (math.isfinite(temp_f) and math.isfinite(rh)):
        raise ValueError("temperature and humidity must be finite numbers")
    if not 0 <= rh <= 100:
        raise ValueError("relative humidity must be between 0 and 100 percent")

    simple = 0.5 * (temp_f + 61.0 + (temp_f - 68.0) * 1.2 + rh * 0.094)
    if (simple + temp_f) / 2 < 80:
        return simple

    t, r = temp_f, rh
    hi = (
        -42.379
        + 2.04901523 * t
        + 10.14333127 * r
        - 0.22475541 * t * r
        - 0.00683783 * t * t
        - 0.05481717 * r * r
        + 0.00122874 * t * t * r
        + 0.00085282 * t * r * r
        - 0.00000199 * t * t * r * r
    )
    if r < 13 and 80 <= t <= 112:
        hi -= ((13 - r) / 4) * math.sqrt((17 - abs(t - 95)) / 17)
    elif r > 85 and 80 <= t <= 87:
        hi += ((r - 85) / 10) * ((87 - t) / 5)
    return hi


def classify(hi_f: float) -> str:
    """Return ``"high"``, ``"initial"`` or ``"below_initial"`` for a heat index."""
    if hi_f >= HIGH_HEAT_TRIGGER_F:
        return HIGH
    if hi_f >= INITIAL_HEAT_TRIGGER_F:
        return INITIAL
    return NONE


def assess(temp_f: float, rh: float) -> Dict[str, object]:
    """Heat index and trigger status for one condition, as a plain dict."""
    hi = heat_index(temp_f, rh)
    return {
        "temp_f": temp_f,
        "rh_percent": rh,
        "heat_index_f": round(hi, 1),
        "trigger": classify(hi),
        "source": OSHA_SOURCE,
        "note": "Proposed rule, not in force. Heat index assumes shade; sun can add up to 15 F.",
    }
