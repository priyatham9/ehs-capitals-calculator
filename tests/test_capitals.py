"""Tests for the human-capital-inclusive cost-benefit model.

The first class pins the model against the six figures published in the source
article. If a refactor breaks those, the model no longer reproduces the work it
claims to implement.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from capitals import (  # noqa: E402
    HUMAN_CAPITAL,
    SOURCE_URL,
    TRADITIONAL,
    CostModel,
    evaluate,
    worked_example,
)


class TestPublishedFigures(unittest.TestCase):
    """Pin to Brandon (2025), LeadingEHS.com - the loading-station example."""

    def setUp(self):
        self.traditional, self.enhanced = worked_example(horizon_years=5)

    def test_traditional_annual_cost_is_55k(self):
        self.assertEqual(self.traditional.annual_cost, 55_000)

    def test_enhanced_annual_cost_is_118k(self):
        self.assertEqual(self.enhanced.annual_cost, 118_000)

    def test_traditional_payback_is_3_41_years(self):
        self.assertAlmostEqual(self.traditional.payback_years, 3.41, places=2)

    def test_enhanced_payback_is_1_59_years(self):
        self.assertAlmostEqual(self.enhanced.payback_years, 1.59, places=2)

    def test_traditional_five_year_roi_is_147_percent(self):
        self.assertAlmostEqual(self.traditional.roi, 1.467, places=3)

    def test_enhanced_five_year_roi_is_315_percent(self):
        self.assertAlmostEqual(self.enhanced.roi, 3.147, places=3)

    def test_human_capital_accounting_more_than_doubles_the_case(self):
        self.assertGreater(self.enhanced.roi, 2 * self.traditional.roi)


class TestModelProperties(unittest.TestCase):
    def test_enhanced_model_is_a_superset_of_traditional(self):
        for key, value in TRADITIONAL.components.items():
            self.assertEqual(HUMAN_CAPITAL.components[key], value)

    def test_counting_more_cost_can_never_weaken_the_case(self):
        """Adding non-negative cost lines must not lengthen payback."""
        base = evaluate(TRADITIONAL, 100_000, 0.5)
        more = evaluate(HUMAN_CAPITAL, 100_000, 0.5)
        self.assertLessEqual(more.payback_years, base.payback_years)

    def test_zero_risk_reduction_never_pays_back(self):
        r = evaluate(TRADITIONAL, 100_000, 0.0)
        self.assertIsNone(r.payback_years)
        self.assertIsNone(r.roi)

    def test_full_risk_reduction_uses_entire_annual_cost(self):
        r = evaluate(TRADITIONAL, 100_000, 1.0)
        self.assertEqual(r.annual_benefit, r.annual_cost)

    def test_payback_scales_linearly_with_capex(self):
        a = evaluate(TRADITIONAL, 100_000, 0.8)
        b = evaluate(TRADITIONAL, 200_000, 0.8)
        self.assertAlmostEqual(b.payback_years, 2 * a.payback_years)

    def test_roi_and_net_roi_differ_by_exactly_one(self):
        r = evaluate(HUMAN_CAPITAL, 150_000, 0.8, horizon_years=5)
        self.assertAlmostEqual(r.roi - r.net_roi, 1.0)


class TestPublishedLineItems(unittest.TestCase):
    """The article publishes the cost table, not just the totals.

    These pin each transcribed line item. If someone "tidies" a number, the
    repo stops reproducing the source it claims to reproduce.
    """

    TRADITIONAL_LINES = {
        "equipment_downtime": 30_000,
        "material_loss_spills": 10_000,
        "injury_costs": 15_000,
    }
    HUMAN_CAPITAL_LINES = {
        "lost_work_time": 20_000,
        "retraining_turnover": 8_000,
        "productivity_loss": 25_000,
        "morale_team_performance": 10_000,
    }

    def test_traditional_line_items_match_the_published_table(self):
        self.assertEqual(dict(TRADITIONAL.components), self.TRADITIONAL_LINES)

    def test_human_capital_line_items_match_the_published_table(self):
        expected = {**self.TRADITIONAL_LINES, **self.HUMAN_CAPITAL_LINES}
        self.assertEqual(dict(HUMAN_CAPITAL.components), expected)

    def test_line_items_sum_to_the_published_totals(self):
        self.assertEqual(TRADITIONAL.annual_cost, 55_000)
        self.assertEqual(HUMAN_CAPITAL.annual_cost, 118_000)

    def test_residual_cost_matches_the_published_table(self):
        """The article shows residual cost at 20%: $11,000 and $23,600."""
        traditional, enhanced = worked_example()
        self.assertAlmostEqual(traditional.residual_cost, 11_000)
        self.assertAlmostEqual(enhanced.residual_cost, 23_600)

    def test_annual_savings_match_the_published_table(self):
        """The article shows annual cost savings of $44,000 and $94,400."""
        traditional, enhanced = worked_example()
        self.assertAlmostEqual(traditional.annual_benefit, 44_000)
        self.assertAlmostEqual(enhanced.annual_benefit, 94_400)

    def test_residual_plus_benefit_reconstructs_annual_cost(self):
        for result in worked_example():
            self.assertAlmostEqual(
                result.residual_cost + result.annual_benefit, result.annual_cost
            )

    def test_source_url_is_recorded(self):
        self.assertIn("leadingehs.com", SOURCE_URL)


class TestJavaScriptParity(unittest.TestCase):
    """Keep docs/index.html and src/capitals.py computing the same thing.

    The fixture was checked against the JavaScript in docs/index.html running
    in a real browser on 3 September 2026; all six cases agreed bit-exactly.
    See tests/parity_cases.json for the full verification note, including what
    a stored double can and cannot establish about which side produced it.

    This is a snapshot, not a live check: nothing here executes the JavaScript,
    so an edit to the JS arithmetic will not fail this test. Regenerate the
    fixture when you change either implementation.
    """

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parent / "parity_cases.json"
        with path.open() as handle:
            cls.cases = json.load(handle)["cases"]

    def test_fixture_is_populated(self):
        self.assertGreaterEqual(len(self.cases), 6)

    def test_python_matches_javascript_on_every_case(self):
        for case in self.cases:
            with self.subTest(case=case["name"]):
                model = CostModel("parity", {"total": float(case["annual_cost"])})
                result = evaluate(
                    model,
                    case["capex"],
                    case["risk_reduction"],
                    case["horizon_years"],
                )
                expected = case["expected"]
                self.assertAlmostEqual(
                    result.residual_cost, expected["residual_cost"], places=6
                )
                for field in ("payback_years", "roi", "net_roi"):
                    actual, want = getattr(result, field), expected[field]
                    if want is None:
                        self.assertIsNone(actual)
                    else:
                        self.assertAlmostEqual(actual, want, places=9)


class TestInputValidation(unittest.TestCase):
    def test_rejects_zero_capex(self):
        with self.assertRaises(ValueError):
            evaluate(TRADITIONAL, 0, 0.8)

    def test_rejects_negative_capex(self):
        with self.assertRaises(ValueError):
            evaluate(TRADITIONAL, -1, 0.8)

    def test_rejects_risk_reduction_above_one(self):
        """Guards the most likely user error: entering 80 instead of 0.80."""
        with self.assertRaises(ValueError):
            evaluate(TRADITIONAL, 150_000, 80)

    def test_rejects_negative_risk_reduction(self):
        with self.assertRaises(ValueError):
            evaluate(TRADITIONAL, 150_000, -0.1)

    def test_rejects_zero_horizon(self):
        with self.assertRaises(ValueError):
            evaluate(TRADITIONAL, 150_000, 0.8, horizon_years=0)

    def test_rejects_negative_cost_component(self):
        bad = CostModel("bad", {"downtime": -5_000.0})
        with self.assertRaises(ValueError):
            evaluate(bad, 150_000, 0.8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
