"""Tests for the deterministic sensitivity analysis.

Where a sensitivity result has a closed form, the test asserts the closed form
rather than a recorded output. A payback hurdle of H years is met exactly when
capex == H * annual_cost * risk_reduction, so every breakeven in this module
can be checked against arithmetic rather than against a previous run.
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from capitals import (  # noqa: E402
    HUMAN_CAPITAL,
    LOADING_STATION_CAPEX,
    LOADING_STATION_RISK_REDUCTION,
    CostModel,
)
from sensitivity import (  # noqa: E402
    COST_PREFIX,
    Scenario,
    decision_sensitivity,
    flip_point,
    format_tornado,
    grid,
    metric_value,
    payback_within,
    relative_ranges,
    tornado,
)


def baseline() -> Scenario:
    return Scenario(
        HUMAN_CAPITAL, LOADING_STATION_CAPEX, LOADING_STATION_RISK_REDUCTION, 5
    )


class TestScenario(unittest.TestCase):
    def test_parameter_names_cover_every_input(self):
        names = baseline().parameter_names()
        self.assertIn("capex", names)
        self.assertIn("risk_reduction", names)
        self.assertIn("horizon_years", names)
        for key in HUMAN_CAPITAL.components:
            self.assertIn(COST_PREFIX + key, names)
        self.assertEqual(len(names), len(HUMAN_CAPITAL.components) + 3)

    def test_get_returns_current_values(self):
        scenario = baseline()
        self.assertEqual(scenario.get("capex"), 150_000)
        self.assertEqual(scenario.get("risk_reduction"), 0.80)
        self.assertEqual(scenario.get(COST_PREFIX + "productivity_loss"), 25_000)

    def test_with_value_does_not_mutate_the_original(self):
        scenario = baseline()
        changed = scenario.with_value("capex", 999_000)
        self.assertEqual(scenario.capex, 150_000)
        self.assertEqual(changed.capex, 999_000)

    def test_with_value_does_not_mutate_the_original_cost_model(self):
        """Guards against sharing the components dict between scenarios."""
        scenario = baseline()
        changed = scenario.with_value(COST_PREFIX + "lost_work_time", 1.0)
        self.assertEqual(scenario.model.components["lost_work_time"], 20_000)
        self.assertEqual(changed.model.components["lost_work_time"], 1.0)

    def test_horizon_is_coerced_to_an_integer(self):
        changed = baseline().with_value("horizon_years", 7.4)
        self.assertEqual(changed.horizon_years, 7)
        self.assertIsInstance(changed.horizon_years, int)

    def test_unknown_parameter_raises(self):
        with self.assertRaises(KeyError):
            baseline().get("discount_rate")
        with self.assertRaises(KeyError):
            baseline().with_value("cost:nonexistent", 1.0)

    def test_evaluate_matches_the_published_example(self):
        self.assertAlmostEqual(baseline().evaluate().payback_years, 1.59, places=2)


class TestMetricValue(unittest.TestCase):
    def test_never_paying_back_maps_to_infinity(self):
        result = baseline().with_value("risk_reduction", 0.0).evaluate()
        self.assertTrue(math.isinf(metric_value(result, "payback_years")))

    def test_never_paying_back_maps_roi_to_zero(self):
        result = baseline().with_value("risk_reduction", 0.0).evaluate()
        self.assertEqual(metric_value(result, "roi"), 0.0)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            metric_value(baseline().evaluate(), "irr")


class TestRelativeRanges(unittest.TestCase):
    def test_bands_are_symmetric_about_the_baseline(self):
        ranges = relative_ranges(baseline(), fraction=0.20)
        low, high = ranges["capex"]
        self.assertAlmostEqual(low, 120_000)
        self.assertAlmostEqual(high, 180_000)

    def test_risk_reduction_is_clamped_to_one(self):
        """0.80 x 1.5 would be 1.20, which the model would reject."""
        _, high = relative_ranges(baseline(), fraction=0.50)["risk_reduction"]
        self.assertLessEqual(high, 1.0)

    def test_risk_reduction_is_clamped_to_zero(self):
        scenario = baseline().with_value("risk_reduction", 0.1)
        low, _ = relative_ranges(scenario, fraction=2.0)["risk_reduction"]
        self.assertGreaterEqual(low, 0.0)

    def test_costs_are_floored_at_zero(self):
        low, _ = relative_ranges(baseline(), fraction=3.0)[
            COST_PREFIX + "injury_costs"
        ]
        self.assertGreaterEqual(low, 0.0)

    def test_horizon_can_be_excluded(self):
        ranges = relative_ranges(baseline(), include_horizon=False)
        self.assertNotIn("horizon_years", ranges)

    def test_negative_fraction_raises(self):
        with self.assertRaises(ValueError):
            relative_ranges(baseline(), fraction=-0.1)


class TestTornado(unittest.TestCase):
    def test_bars_are_sorted_by_swing_descending(self):
        bars = tornado(baseline())
        swings = [b.swing for b in bars]
        self.assertEqual(swings, sorted(swings, reverse=True))

    def test_risk_reduction_dominates_the_published_example(self):
        """The whole point of the tool: the assumption drives the answer."""
        bars = tornado(baseline())
        self.assertEqual(bars[0].parameter, "risk_reduction")

    def test_horizon_has_no_effect_on_payback(self):
        bars = {b.parameter: b for b in tornado(baseline(), metric="payback_years")}
        self.assertEqual(bars["horizon_years"].swing, 0.0)

    def test_horizon_does_affect_roi(self):
        bars = {b.parameter: b for b in tornado(baseline(), metric="roi")}
        self.assertGreater(bars["horizon_years"].swing, 0.0)

    def test_larger_cost_lines_swing_payback_further(self):
        """Under a proportional band, swing must order by line size."""
        bars = {b.parameter: b for b in tornado(baseline())}
        self.assertGreater(
            bars[COST_PREFIX + "productivity_loss"].swing,  # $25,000
            bars[COST_PREFIX + "retraining_turnover"].swing,  # $8,000
        )

    def test_a_zero_width_band_produces_zero_swing(self):
        bars = tornado(baseline(), relative_ranges(baseline(), fraction=0.0))
        self.assertTrue(all(b.swing == 0.0 for b in bars))

    def test_swing_is_infinite_when_one_end_never_pays_back(self):
        scenario = baseline().with_value("risk_reduction", 0.05)
        ranges = {"risk_reduction": (0.0, 0.1)}
        bar = tornado(scenario, ranges)[0]
        self.assertTrue(math.isinf(bar.swing))

    def test_best_and_worst_respect_metric_direction(self):
        bar = {b.parameter: b for b in tornado(baseline())}["capex"]
        self.assertLess(bar.best_output, bar.worst_output)  # lower payback better
        roi_bar = {b.parameter: b for b in tornado(baseline(), metric="roi")}["capex"]
        self.assertGreater(roi_bar.best_output, roi_bar.worst_output)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            tornado(baseline(), metric="irr")

    def test_unknown_parameter_in_ranges_raises(self):
        with self.assertRaises(KeyError):
            tornado(baseline(), {"discount_rate": (0.0, 0.1)})

    def test_ordering_is_deterministic_across_runs(self):
        first = [b.parameter for b in tornado(baseline())]
        second = [b.parameter for b in tornado(baseline())]
        self.assertEqual(first, second)


class TestFlipPoint(unittest.TestCase):
    """Breakeven values have a closed form; assert against it, not a snapshot."""

    def test_capex_flip_matches_the_closed_form(self):
        # Hurdle H is met exactly when capex == H * annual_benefit.
        scenario = baseline()
        hurdle = 2.0
        expected = hurdle * scenario.model.annual_cost * scenario.risk_reduction
        found = flip_point(
            scenario, "capex", payback_within(hurdle), 50_000, 500_000
        )
        self.assertIsNotNone(found)
        self.assertAlmostEqual(found, expected, places=3)
        self.assertAlmostEqual(expected, 188_800.0)

    def test_risk_reduction_flip_matches_the_closed_form(self):
        # rr == capex / (H * annual_cost)
        scenario = baseline()
        hurdle = 2.0
        expected = scenario.capex / (hurdle * scenario.model.annual_cost)
        found = flip_point(
            scenario, "risk_reduction", payback_within(hurdle), 0.0, 1.0
        )
        self.assertIsNotNone(found)
        self.assertAlmostEqual(found, expected, places=6)

    def test_returns_none_when_the_conclusion_never_changes(self):
        # Every capex in this narrow band clears a generous hurdle.
        found = flip_point(
            baseline(), "capex", payback_within(10.0), 100_000, 200_000
        )
        self.assertIsNone(found)

    def test_integer_parameter_is_scanned_not_bisected(self):
        """Horizon is a step function; a fractional answer would be wrong."""
        scenario = baseline()
        found = flip_point(
            scenario,
            "horizon_years",
            lambda r: r.roi is not None and r.roi >= 3.0,
            1,
            12,
        )
        self.assertIsNotNone(found)
        self.assertEqual(found, float(int(found)))

    def test_inverted_bounds_raise(self):
        with self.assertRaises(ValueError):
            flip_point(baseline(), "capex", payback_within(3.0), 200_000, 100_000)

    def test_payback_within_rejects_non_positive_years(self):
        with self.assertRaises(ValueError):
            payback_within(0)


class TestDecisionSensitivity(unittest.TestCase):
    def test_flip_capable_inputs_are_listed_first(self):
        reports = decision_sensitivity(baseline(), payback_within(2.0))
        flippable = [r.can_flip for r in reports]
        self.assertEqual(flippable, sorted(flippable, reverse=True))

    def test_only_capex_and_risk_reduction_flip_a_two_year_hurdle(self):
        """At +/-30%, no single cost line can push payback past 2 years."""
        reports = {
            r.parameter: r for r in decision_sensitivity(baseline(), payback_within(2.0))
        }
        self.assertTrue(reports["capex"].can_flip)
        self.assertTrue(reports["risk_reduction"].can_flip)
        for name in HUMAN_CAPITAL.components:
            self.assertFalse(reports[COST_PREFIX + name].can_flip)

    def test_baseline_verdict_is_reported_consistently(self):
        reports = decision_sensitivity(baseline(), payback_within(2.0))
        self.assertTrue(all(r.baseline_holds for r in reports))

    def test_a_failing_baseline_is_reported_as_failing(self):
        reports = decision_sensitivity(baseline(), payback_within(0.5))
        self.assertTrue(all(not r.baseline_holds for r in reports))


class TestGrid(unittest.TestCase):
    def test_shape_is_rows_by_columns(self):
        table = grid(
            baseline(),
            "capex", [100_000, 150_000, 200_000],
            "risk_reduction", [0.4, 0.6, 0.8, 1.0],
        )
        self.assertEqual(len(table), 4)
        self.assertTrue(all(len(row) == 3 for row in table))

    def test_a_known_cell_matches_hand_arithmetic(self):
        table = grid(
            baseline(), "capex", [150_000], "risk_reduction", [1.0]
        )
        self.assertAlmostEqual(table[0][0], 150_000 / 118_000)

    def test_payback_falls_as_risk_reduction_rises(self):
        table = grid(
            baseline(), "capex", [150_000], "risk_reduction", [0.2, 0.5, 0.9]
        )
        column = [row[0] for row in table]
        self.assertEqual(column, sorted(column, reverse=True))

    def test_unknown_parameter_raises(self):
        with self.assertRaises(KeyError):
            grid(baseline(), "nope", [1], "risk_reduction", [0.5])


class TestFormatting(unittest.TestCase):
    def test_renders_one_line_per_bar_plus_a_header(self):
        bars = tornado(baseline())
        text = format_tornado(bars)
        self.assertEqual(len(text.splitlines()), len(bars) + 1)

    def test_handles_an_empty_bar_list(self):
        self.assertIn("no parameters", format_tornado([]))

    def test_marks_infinite_swings_in_words(self):
        scenario = baseline().with_value("risk_reduction", 0.05)
        bars = tornado(scenario, {"risk_reduction": (0.0, 0.1)})
        self.assertIn("never pays back", format_tornado(bars))


class TestSingleComponentModel(unittest.TestCase):
    """A one-line model must still work; it is the CLI's simplest case."""

    def test_single_component_scenario_evaluates(self):
        scenario = Scenario(CostModel("one", {"total": 100_000.0}), 200_000, 0.5, 5)
        self.assertAlmostEqual(scenario.evaluate().payback_years, 4.0)

    def test_tornado_runs_on_a_single_component_model(self):
        scenario = Scenario(CostModel("one", {"total": 100_000.0}), 200_000, 0.5, 5)
        self.assertEqual(len(tornado(scenario)), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
