"""
Command-line interface to the capitals cost-benefit model.

    python3 src/cli.py example
    python3 src/cli.py compare --capex 200000 --risk-reduction 0.6
    python3 src/cli.py tornado --hurdle 2 --spread 0.4
    python3 src/cli.py montecarlo --samples 100000 --hurdle 3
    python3 src/cli.py uncertainty --spread 0.3 --hurdle 3
    python3 src/cli.py heat --temp-f 96 --rh 65

Every subcommand accepts ``--json`` for machine-readable output, and every
subcommand that involves randomness accepts ``--seed``.

Cost components are supplied as repeated ``name=value`` pairs. ``--cost`` adds
a directly booked line that both accountings count; ``--human-cost`` adds a
human-capital line that only the second accounting counts. With no cost flags
at all, the published worked-example figures are used.
"""

import argparse
import json
import math
import sys
from typing import Dict, List, Optional, Sequence

from capitals import (
    HUMAN_CAPITAL,
    LOADING_STATION_CAPEX,
    LOADING_STATION_RISK_REDUCTION,
    SOURCE_URL,
    TRADITIONAL,
    CostModel,
    Result,
    evaluate,
)
import heat
import uncertainty
from sensitivity import (
    METRICS,
    Scenario,
    decision_sensitivity,
    format_tornado,
    payback_within,
    relative_ranges,
    tornado,
)

# ``montecarlo`` is the only module in this repository that needs numpy, and it
# is imported lazily inside cmd_montecarlo() so that the example, compare and
# tornado subcommands run on a bare standard-library interpreter. These two
# defaults are mirrored here only so that building the parser (and rendering
# --help) does not drag numpy in; TestMonteCarloDefaultsMatch pins them to the
# montecarlo module so the copies cannot drift.
DEFAULT_SEED = 1904
DEFAULT_SAMPLES = 50_000

_ASSUMPTION_NOTE = (
    "Note: risk_reduction is an assumption you supplied. This tool does not "
    "estimate it and cannot tell you whether it is achievable."
)


# --------------------------------------------------------------------------
# Argument helpers
# --------------------------------------------------------------------------


def fail(message: str) -> None:
    """Report a user-input error on stderr and exit with code 2.

    Used for validation that happens after argparse has finished, so that
    every kind of bad input leaves the process the same way.
    """
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def parse_cost_pairs(pairs: Optional[Sequence[str]]) -> Dict[str, float]:
    """Parse repeated ``name=value`` arguments into a component mapping.

    Raises:
        ValueError: on malformed pairs, non-numeric values, or negative
            amounts. Callers are expected to turn this into a clean exit.
    """
    out: Dict[str, float] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"expected name=value, got {pair!r}")
        name, _, raw = pair.partition("=")
        name = name.strip()
        if not name:
            raise ValueError(f"empty component name in {pair!r}")
        try:
            value = float(raw.replace(",", "").replace("$", "").strip())
        except ValueError:
            raise ValueError(f"{raw!r} is not a number (in {pair!r})")
        if value < 0:
            raise ValueError(
                f"cost components must be non-negative, got {value} for {name!r}"
            )
        out[name] = value
    return out


def build_models(args: argparse.Namespace) -> List[CostModel]:
    """Build the traditional and human-capital models from CLI arguments.

    With no cost flags supplied, returns the two published worked-example
    models unchanged.

    Raises:
        SystemExit: on malformed or contradictory cost arguments.
    """
    try:
        booked = parse_cost_pairs(getattr(args, "cost", None))
        human = parse_cost_pairs(getattr(args, "human_cost", None))
    except ValueError as exc:
        fail(str(exc))

    if not booked and not human:
        return [TRADITIONAL, HUMAN_CAPITAL]

    overlap = set(booked) & set(human)
    if overlap:
        fail(
            f"{', '.join(sorted(overlap))} given as both --cost and "
            "--human-cost; each component belongs to exactly one accounting"
        )

    traditional = CostModel("Traditional", dict(booked))
    enhanced = CostModel("Human-capital-inclusive", {**booked, **human})
    return [traditional, enhanced]


def result_to_dict(result: Result) -> Dict[str, object]:
    """Flatten a Result for JSON output. ``None`` survives as JSON null."""
    return {
        "label": result.label,
        "annual_cost": result.annual_cost,
        "annual_benefit": result.annual_benefit,
        "residual_cost": result.residual_cost,
        "capex": result.capex,
        "horizon_years": result.horizon_years,
        "payback_years": result.payback_years,
        "roi": result.roi,
        "net_roi": result.net_roi,
        "total_benefit": result.total_benefit,
    }


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------


def cmd_example(args: argparse.Namespace) -> int:
    """Reproduce the published loading-station comparison verbatim."""
    results = [
        evaluate(m, LOADING_STATION_CAPEX, LOADING_STATION_RISK_REDUCTION, 5)
        for m in (TRADITIONAL, HUMAN_CAPITAL)
    ]
    if args.json:
        print(
            json.dumps(
                {
                    "source": SOURCE_URL,
                    "capex": LOADING_STATION_CAPEX,
                    "risk_reduction": LOADING_STATION_RISK_REDUCTION,
                    "results": [result_to_dict(r) for r in results],
                },
                indent=2,
            )
        )
        return 0

    print("Published worked example: loading-station safety investment")
    print(f"  source            {SOURCE_URL}")
    print(f"  capital cost      ${LOADING_STATION_CAPEX:,.0f}")
    print(f"  risk reduction    {LOADING_STATION_RISK_REDUCTION:.0%}")
    print()
    for result in results:
        print(" ", result)
    print()
    print("  Cost lines counted:")
    for name, value in HUMAN_CAPITAL.components.items():
        mark = "both" if name in TRADITIONAL.components else "human-capital only"
        print(f"    {name:<28} ${value:>9,.0f}   ({mark})")
    return 0


def cmd_heat(args: argparse.Namespace) -> int:
    """NWS heat index and OSHA proposed-rule trigger status for one condition."""
    result = heat.assess(args.temp_f, args.rh)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    labels = {
        heat.NONE: "below the initial heat trigger",
        heat.INITIAL: "at or above the initial heat trigger",
        heat.HIGH: "at or above the high heat trigger",
    }
    print(f"  {args.temp_f:g} F at {args.rh:g}% RH -> heat index {result['heat_index_f']} F")
    print(
        f"  {labels[result['trigger']]} "
        f"(initial {heat.INITIAL_HEAT_TRIGGER_F:g} F, high {heat.HIGH_HEAT_TRIGGER_F:g} F)"
    )
    print(f"  {result['note']}")
    print(f"  Source: {heat.OSHA_SOURCE}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare the two accountings on user-supplied inputs."""
    models = build_models(args)
    results = [
        evaluate(m, args.capex, args.risk_reduction, args.horizon) for m in models
    ]

    if args.json:
        print(json.dumps({"results": [result_to_dict(r) for r in results]}, indent=2))
        return 0

    print(f"  capital cost      ${args.capex:,.0f}")
    print(f"  risk reduction    {args.risk_reduction:.0%}")
    print(f"  horizon           {args.horizon} yr")
    print()
    for result in results:
        print(" ", result)

    if args.hurdle is not None:
        print()
        print(f"  Against a {args.hurdle:g}-year payback hurdle:")
        for result in results:
            if result.payback_years is None:
                verdict = "never pays back"
            elif result.payback_years <= args.hurdle:
                verdict = "passes"
            else:
                verdict = "fails"
            print(f"    {result.label:<26} {verdict}")
    print()
    print(f"  {_ASSUMPTION_NOTE}")
    return 0


def cmd_tornado(args: argparse.Namespace) -> int:
    """Rank inputs by how much they move the answer."""
    models = build_models(args)
    model = models[-1] if args.model == "human-capital" else models[0]
    scenario = Scenario(model, args.capex, args.risk_reduction, args.horizon)
    ranges = relative_ranges(scenario, fraction=args.spread)
    bars = tornado(scenario, ranges, metric=args.metric)
    flips = decision_sensitivity(scenario, payback_within(args.hurdle), ranges)
    baseline = scenario.evaluate()

    if args.json:
        print(
            json.dumps(
                {
                    "model": model.label,
                    "metric": args.metric,
                    "spread": args.spread,
                    "hurdle_years": args.hurdle,
                    "baseline": result_to_dict(baseline),
                    "bars": [
                        {
                            "parameter": b.parameter,
                            "low_input": b.low_input,
                            "high_input": b.high_input,
                            "low_output": _jsonable(b.low_output),
                            "high_output": _jsonable(b.high_output),
                            "swing": _jsonable(b.swing),
                        }
                        for b in bars
                    ],
                    "flips": [
                        {
                            "parameter": f.parameter,
                            "can_flip": f.can_flip,
                            "flip_input": f.flip_input,
                        }
                        for f in flips
                    ],
                },
                indent=2,
            )
        )
        return 0

    print(f"{model.label}  (inputs varied +/-{args.spread:.0%})")
    print(f"  baseline: {baseline}")
    print()
    print(format_tornado(bars))
    print()
    print(f"Can one input alone break the {args.hurdle:g}-year payback hurdle?")
    baseline_holds = flips[0].baseline_holds if flips else False
    print(f"  baseline currently {'passes' if baseline_holds else 'fails'}")
    movers = [f for f in flips if f.can_flip]
    if not movers:
        print(f"  No single input, moved +/-{args.spread:.0%}, changes the verdict.")
    for report in movers:
        print(
            f"  {report.parameter:<28} verdict flips at "
            f"{_human(report.flip_input)}  "
            f"(baseline {_human(report.baseline_input)})"
        )
    print()
    print(f"  {_ASSUMPTION_NOTE}")
    return 0


def _parse_ranges(pairs: Optional[Sequence[str]]) -> Dict[str, tuple]:
    """Parse repeated ``NAME=LOW:HIGH`` arguments into range overrides."""
    out: Dict[str, tuple] = {}
    for pair in pairs or []:
        name, sep, raw = pair.partition("=")
        low, colon, high = raw.partition(":")
        if not (sep and colon and name.strip()):
            raise ValueError(f"expected NAME=LOW:HIGH, got {pair!r}")
        try:
            out[name.strip()] = (float(low), float(high))
        except ValueError:
            raise ValueError(f"LOW and HIGH must be numbers (in {pair!r})")
    return out


def cmd_uncertainty(args: argparse.Namespace) -> int:
    """Exact interval bounds plus seeded uniform sampling over assumed ranges."""
    if args.spread < 0:
        fail("--spread must be non-negative")
    overrides = _parse_ranges(args.range)
    models = build_models(args)
    known = {n for m in models for n in Scenario(m, 1.0, 0.5).parameter_names()}
    unknown = sorted(set(overrides) - known)
    if unknown:
        fail(f"unknown --range parameter(s) {unknown}; expected one of {sorted(known)}")

    outputs = []
    for model in models:
        scenario = Scenario(model, args.capex, args.risk_reduction, args.horizon)
        own = {k: v for k, v in overrides.items() if k in scenario.parameter_names()}
        ranges = uncertainty.build_ranges(scenario, args.spread, own)
        bounds = uncertainty.interval_bounds(scenario, ranges)
        sim = uncertainty.simulate(scenario, ranges, args.hurdle, args.samples, args.seed)
        outputs.append((model, scenario.evaluate(), bounds, sim))

    if args.json:
        print(
            json.dumps(
                {
                    "ranges_are_assumptions": True,
                    "spread": args.spread,
                    "range_overrides": {k: list(v) for k, v in overrides.items()},
                    "hurdle_years": args.hurdle,
                    "results": [
                        {
                            "label": m.label,
                            "baseline": result_to_dict(base),
                            "bounds": {
                                k: {"low": _jsonable(v.low), "high": _jsonable(v.high)}
                                for k, v in vars(b).items()
                            },
                            "hurdle_verdict": uncertainty.hurdle_verdict(b, args.hurdle),
                            "simulation": {
                                k: _jsonable(v) if isinstance(v, float) else v
                                for k, v in vars(sim).items()
                            },
                        }
                        for m, base, b, sim in outputs
                    ],
                },
                indent=2,
            )
        )
        return 0

    print(
        f"Uncertainty propagation: inputs within +/-{args.spread:.0%} of baseline "
        "(ASSUMED ranges)"
    )
    for name, (low, high) in overrides.items():
        print(f"  override: {name} in [{low:g}, {high:g}]")
    for model, base, b, sim in outputs:
        print()
        print(f"{model.label}")
        print(f"  baseline          {base}")
        print(
            f"  payback bounds    {_bound(b.payback_years, '{:.2f}')} yr   "
            f"(exact, any joint behaviour inside the ranges)"
        )
        print(f"  net ROI bounds    {_bound(b.net_roi, '{:.0%}')}")
        print(f"  {str(args.hurdle) + "-yr hurdle":<17} {uncertainty.hurdle_verdict(b, args.hurdle)}")
        print(
            f"  uniform sampling  {sim.samples:,} draws, seed {sim.seed}: "
            f"P(payback <= {args.hurdle:g} yr) = {sim.p_within_hurdle:.1%} "
            f"(+/-{sim.standard_error:.1%} s.e.); payback p5/p50/p95 = "
            f"{sim.payback_p05:.2f}/{sim.payback_p50:.2f}/{sim.payback_p95:.2f} yr"
        )
    print()
    print(
        "  ASSUMPTION: the ranges are not measured. They are a uniform relative\n"
        "  band (or your --range overrides); this repository holds no data on how\n"
        "  uncertain any input is. Bounds are exact for those ranges only. Sampling\n"
        "  is uniform and independent, so its probability is a statement about\n"
        "  those assumptions and understates spread when cost lines move together."
    )
    return 0


def _bound(interval, fmt: str) -> str:
    def one(x: float) -> str:
        return "never" if math.isinf(x) else fmt.format(x)

    return f"[{one(interval.low)}, {one(interval.high)}]"


def cmd_montecarlo(args: argparse.Namespace) -> int:
    """Propagate input uncertainty and report payback probabilities.

    Imports :mod:`montecarlo` (and therefore numpy) lazily, so that a missing
    numpy only affects this subcommand rather than the whole CLI.
    """
    try:
        from montecarlo import (
            format_result,
            illustrative_spec,
            run as run_montecarlo,
        )
    except ImportError as exc:  # pragma: no cover - depends on the environment
        print(
            f"The montecarlo subcommand requires numpy ({exc}).\n"
            "Install it with: python3 -m pip install numpy\n"
            "The example, compare and tornado subcommands do not need it.",
            file=sys.stderr,
        )
        return 1

    models = build_models(args)
    outputs = []
    for model in models:
        spec = illustrative_spec(
            model,
            args.capex,
            args.risk_reduction,
            horizon_years=args.horizon,
            cost_spread=args.cost_spread,
            capex_spread=args.capex_spread,
            risk_concentration=args.risk_concentration,
        )
        outputs.append((model, run_montecarlo(spec, args.samples, args.seed)))

    if args.json:
        print(
            json.dumps(
                {
                    "samples": args.samples,
                    "seed": args.seed,
                    "hurdle_years": args.hurdle,
                    "spreads_are_illustrative": True,
                    "results": [
                        dict(label=m.label, **r.summary(args.hurdle))
                        for m, r in outputs
                    ],
                },
                indent=2,
            )
        )
        return 0

    print(
        f"Monte Carlo: {args.samples:,} draws, seed {args.seed}, "
        f"horizon {args.horizon} yr"
    )
    print()
    for _, result in outputs:
        print(format_result(result, hurdle_years=args.hurdle))
        print()
    print(
        "  The spreads used here are illustrative placeholders, not measured\n"
        "  uncertainty. Inputs are sampled independently, which understates the\n"
        "  true spread when cost lines move together. See README."
    )
    return 0


def _human(x: float) -> str:
    """Format an input value readably: thousands separated, small ones precise."""
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:,.4g}"


def _jsonable(x: float) -> Optional[float]:
    """Map non-finite floats to null, which JSON can represent."""
    return None if math.isinf(x) or math.isnan(x) else x


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _proportion(text: str) -> float:
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(
            "must be a proportion in [0, 1] -- use 0.8, not 80"
        )
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def add_common_inputs(parser: argparse.ArgumentParser) -> None:
    """Attach the model-input flags shared by most subcommands."""
    parser.add_argument(
        "--capex",
        type=_positive_float,
        default=LOADING_STATION_CAPEX,
        help="up-front capital investment (default: %(default)s)",
    )
    parser.add_argument(
        "--risk-reduction",
        type=_proportion,
        default=LOADING_STATION_RISK_REDUCTION,
        metavar="P",
        help="assumed proportional reduction in incident cost, 0-1 "
        "(default: %(default)s). An assumption you supply.",
    )
    parser.add_argument(
        "--horizon",
        type=_positive_int,
        default=5,
        help="evaluation period in years (default: %(default)s)",
    )
    parser.add_argument(
        "--cost",
        action="append",
        metavar="NAME=VALUE",
        help="a directly booked annual cost line, counted by both accountings; "
        "repeatable",
    )
    parser.add_argument(
        "--human-cost",
        action="append",
        metavar="NAME=VALUE",
        help="an annual human-capital cost line, counted only by the "
        "human-capital accounting; repeatable",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit JSON instead of text"
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser."""
    parser = argparse.ArgumentParser(
        prog="ehs-capitals",
        description=(
            "Compare a traditional and a human-capital-inclusive accounting of "
            "the same EHS investment."
        ),
        epilog=_ASSUMPTION_NOTE,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    example = subparsers.add_parser(
        "example", help="reproduce the published worked example"
    )
    example.add_argument("--json", action="store_true", help="emit JSON")
    example.set_defaults(func=cmd_example)

    compare = subparsers.add_parser(
        "compare", help="compare both accountings on your own inputs"
    )
    add_common_inputs(compare)
    compare.add_argument(
        "--hurdle",
        type=_positive_float,
        default=None,
        metavar="YEARS",
        help="payback hurdle to judge each accounting against",
    )
    compare.set_defaults(func=cmd_compare)

    torn = subparsers.add_parser(
        "tornado", help="rank inputs by how much they move the answer"
    )
    add_common_inputs(torn)
    torn.add_argument(
        "--model",
        choices=("traditional", "human-capital"),
        default="human-capital",
        help="which accounting to analyse",
    )
    torn.add_argument(
        "--metric", choices=METRICS, default="payback_years", help="output metric"
    )
    torn.add_argument(
        "--spread",
        type=float,
        default=0.30,
        help="relative half-width applied to every input",
    )
    torn.add_argument(
        "--hurdle",
        type=_positive_float,
        default=3.0,
        metavar="YEARS",
        help="payback hurdle used for the decision-flip search",
    )
    torn.set_defaults(func=cmd_tornado)

    mc = subparsers.add_parser(
        "montecarlo", help="propagate input uncertainty into payback probability"
    )
    add_common_inputs(mc)
    mc.add_argument(
        "--samples", type=_positive_int, default=DEFAULT_SAMPLES, help="draws"
    )
    mc.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="RNG seed (fixed by default)"
    )
    mc.add_argument(
        "--hurdle",
        type=_positive_float,
        default=3.0,
        metavar="YEARS",
        help="payback hurdle whose probability is reported",
    )
    mc.add_argument(
        "--cost-spread",
        type=float,
        default=0.30,
        help="illustrative half-width on each cost line",
    )
    mc.add_argument(
        "--capex-spread",
        type=float,
        default=0.15,
        help="illustrative downside half-width on capex; upside is twice this",
    )
    mc.add_argument(
        "--risk-concentration",
        type=float,
        default=20.0,
        help="Beta concentration on risk reduction; higher is a tighter belief",
    )
    mc.set_defaults(func=cmd_montecarlo)

    un = subparsers.add_parser(
        "uncertainty",
        help="exact interval bounds and seeded sampling over assumed input ranges",
        description=(
            "Propagate assumed input ranges through the model. Reports exact "
            "payback and ROI bounds (distribution-free) and a seeded uniform "
            "sample. Standard library only. The ranges are assumptions."
        ),
    )
    add_common_inputs(un)
    un.add_argument(
        "--spread",
        type=float,
        default=0.30,
        help="ASSUMED relative half-width applied to every cost line, capex and "
        "risk reduction",
    )
    un.add_argument(
        "--range",
        action="append",
        metavar="NAME=LOW:HIGH",
        help="ASSUMED absolute range overriding --spread for one input, e.g. "
        "risk_reduction=0.6:0.9 or cost:lost_work_time=10000:30000; repeatable",
    )
    un.add_argument(
        "--samples", type=_positive_int, default=uncertainty.DEFAULT_SAMPLES, help="draws"
    )
    un.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed")
    un.add_argument(
        "--hurdle",
        type=_positive_float,
        default=3.0,
        metavar="YEARS",
        help="payback hurdle whose verdict and probability are reported",
    )
    un.set_defaults(func=cmd_uncertainty)

    ht = subparsers.add_parser(
        "heat",
        help="NWS heat index and OSHA proposed heat-trigger status",
        description=(
            "Compute the NWS heat index from air temperature and relative humidity "
            "and compare it with the initial (80 F) and high (90 F) heat triggers "
            "in OSHA's proposed heat rule (89 FR 70698). The rule is a proposal."
        ),
    )
    ht.add_argument("--temp-f", type=float, required=True, help="air temperature, deg F")
    ht.add_argument("--rh", type=float, required=True, help="relative humidity, percent")
    ht.add_argument("--json", action="store_true", help="machine-readable output")
    ht.set_defaults(func=cmd_heat)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns a process exit code.

    Raises nothing on ordinary bad input: validation failures are reported to
    stderr and returned as exit code 2.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
