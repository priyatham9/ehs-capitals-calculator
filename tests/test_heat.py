"""Tests for the heat index and OSHA proposed-trigger module.

Anchors: the NWS page https://www.weather.gov/safety/heat-index states that
96 F at 65 % RH gives a heat index of 121 F; the OSHA proposed text (89 FR
70698, proposed 29 CFR 1910.148(b)) sets triggers at heat index 80 F and 90 F.
Other values are checked against the published equations written out
independently in the test.
"""

import io
import json
import math
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cli  # noqa: E402
import heat  # noqa: E402


def rothfusz(t, r):
    c = [-42.379, 2.04901523, 10.14333127, -0.22475541, -0.00683783,
         -0.05481717, 0.00122874, 0.00085282, -0.00000199]
    return (c[0] + c[1] * t + c[2] * r + c[3] * t * r + c[4] * t * t + c[5] * r * r
            + c[6] * t * t * r + c[7] * t * r * r + c[8] * t * t * r * r)


class TestHeatIndex(unittest.TestCase):
    def test_nws_published_example(self):
        self.assertEqual(round(heat.heat_index(96, 65)), 121)

    def test_matches_plain_regression_without_adjustment(self):
        self.assertAlmostEqual(heat.heat_index(90, 70), rothfusz(90, 70), places=9)

    def test_low_humidity_adjustment(self):
        t, r = 95, 10
        expected = rothfusz(t, r) - ((13 - r) / 4) * math.sqrt((17 - abs(t - 95)) / 17)
        self.assertAlmostEqual(heat.heat_index(t, r), expected, places=9)
        self.assertLess(heat.heat_index(t, r), rothfusz(t, r))

    def test_high_humidity_adjustment(self):
        t, r = 85, 90
        expected = rothfusz(t, r) + ((r - 85) / 10) * ((87 - t) / 5)
        self.assertAlmostEqual(heat.heat_index(t, r), expected, places=9)

    def test_no_adjustment_outside_documented_ranges(self):
        # RH < 13 but T > 112, and RH > 85 but T > 87: regression only.
        self.assertAlmostEqual(heat.heat_index(113, 10), rothfusz(113, 10), places=9)
        self.assertAlmostEqual(heat.heat_index(90, 90), rothfusz(90, 90), places=9)

    def test_cool_conditions_use_simple_formula(self):
        t, r = 70, 50
        simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + r * 0.094)
        self.assertAlmostEqual(heat.heat_index(t, r), simple, places=12)

    def test_increases_with_humidity_in_hot_conditions(self):
        values = [heat.heat_index(95, r) for r in (30, 40, 50, 60, 70)]
        self.assertEqual(values, sorted(values))

    def test_rejects_bad_input(self):
        for t, r in [(90, -1), (90, 101), (float("nan"), 50), (90, float("inf"))]:
            with self.subTest(t=t, r=r), self.assertRaises(ValueError):
                heat.heat_index(t, r)


class TestTriggers(unittest.TestCase):
    def test_constants_match_proposed_rule(self):
        self.assertEqual(heat.INITIAL_HEAT_TRIGGER_F, 80.0)
        self.assertEqual(heat.HIGH_HEAT_TRIGGER_F, 90.0)
        self.assertEqual(heat.INDOOR_WARNING_SIGN_F, 120.0)

    def test_boundaries_are_inclusive(self):
        self.assertEqual(heat.classify(79.99), heat.NONE)
        self.assertEqual(heat.classify(80.0), heat.INITIAL)
        self.assertEqual(heat.classify(89.99), heat.INITIAL)
        self.assertEqual(heat.classify(90.0), heat.HIGH)
        self.assertEqual(heat.classify(121.0), heat.HIGH)

    def test_assess(self):
        out = heat.assess(96, 65)
        self.assertEqual(out["heat_index_f"], 121.0)
        self.assertEqual(out["trigger"], heat.HIGH)
        self.assertIn("89 FR 70698", out["source"])
        self.assertEqual(heat.assess(70, 50)["trigger"], heat.NONE)


class TestCli(unittest.TestCase):
    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_json(self):
        code, out, _ = self.run_cli(["heat", "--temp-f", "96", "--rh", "65", "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["trigger"], "high")

    def test_text(self):
        code, out, _ = self.run_cli(["heat", "--temp-f", "85", "--rh", "40"])
        self.assertEqual(code, 0)
        self.assertIn("heat index", out)
        self.assertIn("89 FR 70698", out)

    def test_bad_humidity_exits_2(self):
        code, _, err = self.run_cli(["heat", "--temp-f", "90", "--rh", "150"])
        self.assertEqual(code, 2)
        self.assertIn("error", err)


if __name__ == "__main__":
    unittest.main()
