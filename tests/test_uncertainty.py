"""Tests for interval and seeded-sampling uncertainty propagation."""

import io
import json
import math
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cli  # noqa: E402
import uncertainty  # noqa: E402
from capitals import HUMAN_CAPITAL, TRADITIONAL  # noqa: E402
from sensitivity import Scenario  # noqa: E402


def scen(model=HUMAN_CAPITAL):
    return Scenario(model, 150_000.0, 0.8, 5)


def capture(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class TestBounds(unittest.TestCase):
    def test_zero_spread_collapses_to_the_published_point(self):
        s = scen(TRADITIONAL)
        b = uncertainty.interval_bounds(s, uncertainty.build_ranges(s, 0.0))
        self.assertAlmostEqual(b.payback_years.low, 150_000 / 44_000)
        self.assertAlmostEqual(b.payback_years.high, 150_000 / 44_000)
        self.assertAlmostEqual(b.net_roi.low, 0.4666666667)

    def test_closed_form_payback_bounds(self):
        s = scen(TRADITIONAL)
        r = uncertainty.build_ranges(s, 0.1)
        b = uncertainty.interval_bounds(s, r)
        # costs +/-10% (55,000 total), capex +/-10%, risk reduction 0.72..0.88
        self.assertAlmostEqual(b.payback_years.low, 135_000 / (60_500 * 0.88))
        self.assertAlmostEqual(b.payback_years.high, 165_000 / (49_500 * 0.72))

    def test_bounds_match_brute_force_corners_of_the_model(self):
        for model in (TRADITIONAL, HUMAN_CAPITAL):
            s = scen(model)
            r = uncertainty.build_ranges(s, 0.25)
            b = uncertainty.interval_bounds(s, r)
            lo, hi = uncertainty.corner_check(s, r)
            self.assertAlmostEqual(b.payback_years.low, lo)
            self.assertAlmostEqual(b.payback_years.high, hi)

    def test_zero_benefit_corner_gives_unbounded_payback(self):
        s = scen(TRADITIONAL)
        r = uncertainty.build_ranges(s, 0.0, {"risk_reduction": (0.0, 0.5)})
        self.assertTrue(math.isinf(uncertainty.interval_bounds(s, r).payback_years.high))

    def test_override_wins_over_spread(self):
        s = scen()
        r = uncertainty.build_ranges(s, 0.3, {"capex": (100_000, 120_000)})
        self.assertEqual(r["capex"], (100_000.0, 120_000.0))

    def test_horizon_fixed_unless_given(self):
        s = scen()
        self.assertEqual(uncertainty.build_ranges(s, 0.3)["horizon_years"], (5.0, 5.0))

    def test_invalid_ranges_rejected(self):
        s = scen()
        for bad in (
            {"capex": (0, 10)},
            {"capex": (10, 5)},
            {"risk_reduction": (0.5, 1.2)},
            {"nope": (1, 2)},
            {"cost:injury_costs": (-1, 5)},
        ):
            with self.assertRaises(ValueError):
                uncertainty.build_ranges(s, 0.1, bad)

    def test_verdicts(self):
        s = scen()
        b = uncertainty.interval_bounds(s, uncertainty.build_ranges(s, 0.0))
        self.assertIn("passes", uncertainty.hurdle_verdict(b, 3.0))
        self.assertIn("fails", uncertainty.hurdle_verdict(b, 1.0))
        b = uncertainty.interval_bounds(s, uncertainty.build_ranges(s, 0.3))
        self.assertIn("depends", uncertainty.hurdle_verdict(b, 1.5))


class TestSimulation(unittest.TestCase):
    def test_same_seed_reproduces_exactly(self):
        s = scen()
        r = uncertainty.build_ranges(s, 0.3)
        a = uncertainty.simulate(s, r, 3.0, 2000, 7)
        b = uncertainty.simulate(s, r, 3.0, 2000, 7)
        self.assertEqual(a, b)

    def test_different_seed_differs(self):
        s = scen()
        r = uncertainty.build_ranges(s, 0.3)
        self.assertNotEqual(
            uncertainty.simulate(s, r, 3.0, 2000, 1).payback_p50,
            uncertainty.simulate(s, r, 3.0, 2000, 2).payback_p50,
        )

    def test_percentiles_lie_inside_exact_bounds_and_are_ordered(self):
        s = scen(TRADITIONAL)
        r = uncertainty.build_ranges(s, 0.3)
        b = uncertainty.interval_bounds(s, r)
        m = uncertainty.simulate(s, r, 3.0, 5000, 3)
        self.assertLessEqual(m.payback_p05, m.payback_p50)
        self.assertLessEqual(m.payback_p50, m.payback_p95)
        self.assertGreaterEqual(m.payback_p05, b.payback_years.low)
        self.assertLessEqual(m.payback_p95, b.payback_years.high)
        self.assertGreaterEqual(m.net_roi_p05, b.net_roi.low)
        self.assertLessEqual(m.net_roi_p95, b.net_roi.high)

    def test_degenerate_box_gives_certain_answer(self):
        s = scen()
        m = uncertainty.simulate(s, uncertainty.build_ranges(s, 0.0), 3.0, 50, 1)
        self.assertEqual(m.p_within_hurdle, 1.0)
        self.assertAlmostEqual(m.payback_p50, 150_000 / 94_400)

    def test_invalid_samples(self):
        s = scen()
        with self.assertRaises(ValueError):
            uncertainty.simulate(s, uncertainty.build_ranges(s, 0.1), samples=0)


class TestCli(unittest.TestCase):
    def test_runs_and_labels_assumptions(self):
        code, out, _ = capture(["uncertainty", "--samples", "500"])
        self.assertEqual(code, 0)
        self.assertIn("ASSUMED", out)
        self.assertIn("ASSUMPTION", out)

    def test_seeded_output_is_reproducible(self):
        argv = ["uncertainty", "--samples", "1000", "--seed", "5"]
        self.assertEqual(capture(argv)[1], capture(argv)[1])

    def test_json_flags_assumptions_and_bounds(self):
        _, out, _ = capture(["uncertainty", "--samples", "500", "--json"])
        payload = json.loads(out)
        self.assertTrue(payload["ranges_are_assumptions"])
        trad = payload["results"][0]
        self.assertLessEqual(
            trad["bounds"]["payback_years"]["low"], trad["baseline"]["payback_years"]
        )
        self.assertGreaterEqual(
            trad["bounds"]["payback_years"]["high"], trad["baseline"]["payback_years"]
        )

    def test_range_override_applies(self):
        _, out, _ = capture(
            ["uncertainty", "--samples", "200", "--json", "--spread", "0",
             "--range", "risk_reduction=0.5:0.5"]
        )
        res = json.loads(out)["results"][0]
        self.assertAlmostEqual(res["bounds"]["payback_years"]["low"], 150_000 / 27_500)

    def test_bad_range_exits_2(self):
        for bad in ("risk_reduction", "risk_reduction=a:b", "bogus=1:2", "capex=5:1"):
            err = io.StringIO()
            try:
                with redirect_stderr(err), redirect_stdout(io.StringIO()):
                    code = cli.main(["uncertainty", "--samples", "10", "--range", bad])
            except SystemExit as exc:  # fail() exits directly
                code = exc.code
            self.assertEqual(code, 2, bad)
            self.assertIn("error", err.getvalue())

    def test_does_not_need_numpy(self):
        src = (Path(uncertainty.__file__)).read_text(encoding="utf-8")
        self.assertNotIn("numpy", src.replace("numpy, shaped", ""))

    def test_seed_default_matches_montecarlo(self):
        import montecarlo

        self.assertEqual(uncertainty.DEFAULT_SEED, montecarlo.DEFAULT_SEED)


if __name__ == "__main__":
    unittest.main()
