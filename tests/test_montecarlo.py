"""Tests for the Monte Carlo module.

Two things are worth testing hard here and are easy to get wrong:

  Reproducibility. A simulation that cannot be re-run to the same numbers is
  not evidence of anything. Same seed must mean identical arrays.

  Agreement with the deterministic model. montecarlo.run() re-implements the
  formulas in vectorised numpy rather than calling capitals.evaluate() in a
  loop. Degenerate (Fixed) distributions collapse the simulation onto a single
  point, where the two implementations must agree exactly.

Statistical assertions use wide tolerances and a fixed seed, so they check that
the machinery is wired correctly without becoming flaky.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from capitals import (  # noqa: E402
    HUMAN_CAPITAL,
    LOADING_STATION_CAPEX,
    LOADING_STATION_RISK_REDUCTION,
    TRADITIONAL,
    CostModel,
    evaluate,
)
from montecarlo import (  # noqa: E402
    DEFAULT_SAMPLES,
    DEFAULT_SEED,
    BoundedBeta,
    Fixed,
    MonteCarloSpec,
    Normal,
    PERT,
    Triangular,
    Uniform,
    format_result,
    illustrative_spec,
    run,
)


def fixed_spec(annual_cost=118_000.0, capex=150_000.0, rr=0.8, horizon=5):
    """A spec with no uncertainty at all, for exact comparison."""
    return MonteCarloSpec(
        label="degenerate",
        components={"total": Fixed(annual_cost)},
        capex=Fixed(capex),
        risk_reduction=Fixed(rr),
        horizon_years=horizon,
    )


class TestDistributions(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_fixed_returns_the_constant(self):
        draws = Fixed(42.0).sample(self.rng, 100)
        self.assertTrue(np.all(draws == 42.0))

    def test_uniform_stays_inside_its_bounds(self):
        draws = Uniform(10.0, 20.0).sample(self.rng, 5_000)
        self.assertGreaterEqual(draws.min(), 10.0)
        self.assertLessEqual(draws.max(), 20.0)

    def test_triangular_stays_inside_its_bounds(self):
        draws = Triangular(1.0, 3.0, 10.0).sample(self.rng, 5_000)
        self.assertGreaterEqual(draws.min(), 1.0)
        self.assertLessEqual(draws.max(), 10.0)

    def test_triangular_handles_a_degenerate_range(self):
        draws = Triangular(5.0, 5.0, 5.0).sample(self.rng, 10)
        self.assertTrue(np.all(draws == 5.0))

    def test_pert_stays_inside_its_bounds(self):
        draws = PERT(0.0, 4.0, 10.0).sample(self.rng, 5_000)
        self.assertGreaterEqual(draws.min(), 0.0)
        self.assertLessEqual(draws.max(), 10.0)

    def test_pert_concentrates_near_the_mode(self):
        """PERT should put more mass near the mode than a uniform would."""
        draws = PERT(0.0, 5.0, 10.0).sample(self.rng, 20_000)
        self.assertAlmostEqual(float(draws.mean()), 5.0, delta=0.15)
        self.assertLess(float(draws.std()), 10.0 / math.sqrt(12))

    def test_pert_mean_estimate_matches_the_standard_formula(self):
        self.assertAlmostEqual(PERT(0.0, 4.0, 10.0).mean_estimate, 26.0 / 6.0)

    def test_pert_handles_a_degenerate_range(self):
        draws = PERT(7.0, 7.0, 7.0).sample(self.rng, 10)
        self.assertTrue(np.all(draws == 7.0))

    def test_normal_truncation_respects_the_lower_bound(self):
        draws = Normal(mean=1.0, sd=5.0, lower=0.0).sample(self.rng, 20_000)
        self.assertGreaterEqual(draws.min(), 0.0)

    def test_normal_with_zero_sd_is_a_constant(self):
        draws = Normal(mean=3.0, sd=0.0).sample(self.rng, 10)
        self.assertTrue(np.all(draws == 3.0))

    def test_normal_raises_when_bounds_exclude_the_mass(self):
        impossible = Normal(mean=1000.0, sd=1.0, lower=0.0, upper=1.0)
        with self.assertRaises(ValueError):
            impossible.sample(self.rng, 100)

    def test_bounded_beta_stays_in_the_unit_interval(self):
        draws = BoundedBeta(mean=0.8, concentration=20.0).sample(self.rng, 20_000)
        self.assertGreaterEqual(draws.min(), 0.0)
        self.assertLessEqual(draws.max(), 1.0)

    def test_bounded_beta_centres_on_its_mean(self):
        draws = BoundedBeta(mean=0.8, concentration=50.0).sample(self.rng, 40_000)
        self.assertAlmostEqual(float(draws.mean()), 0.8, delta=0.01)

    def test_higher_concentration_is_a_tighter_belief(self):
        loose = BoundedBeta(0.8, 5.0).sample(self.rng, 20_000).std()
        tight = BoundedBeta(0.8, 200.0).sample(self.rng, 20_000).std()
        self.assertLess(tight, loose)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            Uniform(10.0, 1.0)
        with self.assertRaises(ValueError):
            Triangular(1.0, 0.0, 2.0)
        with self.assertRaises(ValueError):
            PERT(0.0, 1.0, 2.0, lam=0.0)
        with self.assertRaises(ValueError):
            Normal(mean=1.0, sd=-1.0)
        with self.assertRaises(ValueError):
            BoundedBeta(mean=0.0)
        with self.assertRaises(ValueError):
            BoundedBeta(mean=1.0)
        with self.assertRaises(ValueError):
            BoundedBeta(mean=0.5, concentration=0.0)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_gives_identical_draws(self):
        spec = illustrative_spec(HUMAN_CAPITAL, 150_000, 0.8)
        a = run(spec, samples=2_000, seed=123)
        b = run(spec, samples=2_000, seed=123)
        np.testing.assert_array_equal(a.payback_years, b.payback_years)
        np.testing.assert_array_equal(a.annual_cost, b.annual_cost)

    def test_different_seeds_give_different_draws(self):
        spec = illustrative_spec(HUMAN_CAPITAL, 150_000, 0.8)
        a = run(spec, samples=2_000, seed=1)
        b = run(spec, samples=2_000, seed=2)
        self.assertFalse(np.array_equal(a.payback_years, b.payback_years))

    def test_default_seed_is_recorded_in_the_result(self):
        result = run(fixed_spec(), samples=10)
        self.assertEqual(result.seed, DEFAULT_SEED)

    def test_component_order_does_not_affect_the_draws(self):
        """Components are sampled in sorted order, so dict order is irrelevant."""
        a = MonteCarloSpec(
            "x",
            {"alpha": PERT(1.0, 2.0, 3.0), "beta": PERT(4.0, 5.0, 6.0)},
            Fixed(100.0), Fixed(0.5), 5,
        )
        b = MonteCarloSpec(
            "x",
            {"beta": PERT(4.0, 5.0, 6.0), "alpha": PERT(1.0, 2.0, 3.0)},
            Fixed(100.0), Fixed(0.5), 5,
        )
        np.testing.assert_array_equal(
            run(a, samples=500).annual_cost, run(b, samples=500).annual_cost
        )


class TestAgreementWithDeterministicModel(unittest.TestCase):
    """Degenerate distributions must reproduce capitals.evaluate() exactly."""

    def test_payback_matches_the_deterministic_model(self):
        result = run(fixed_spec(), samples=200)
        expected = evaluate(
            CostModel("x", {"total": 118_000.0}), 150_000.0, 0.8, 5
        )
        self.assertTrue(np.allclose(result.payback_years, expected.payback_years))

    def test_roi_matches_the_deterministic_model(self):
        result = run(fixed_spec(), samples=200)
        expected = evaluate(
            CostModel("x", {"total": 118_000.0}), 150_000.0, 0.8, 5
        )
        self.assertTrue(np.allclose(result.roi, expected.roi))
        self.assertTrue(np.allclose(result.net_roi, expected.net_roi))

    def test_reproduces_the_published_figures(self):
        result = run(fixed_spec(), samples=100)
        self.assertAlmostEqual(float(result.payback_years[0]), 1.59, places=2)
        self.assertAlmostEqual(float(result.roi[0]), 3.147, places=3)

    def test_roi_and_net_roi_differ_by_exactly_one(self):
        result = run(illustrative_spec(HUMAN_CAPITAL, 150_000, 0.8), samples=1_000)
        self.assertTrue(np.allclose(result.roi - result.net_roi, 1.0))

    def test_components_sum_into_annual_cost(self):
        spec = MonteCarloSpec(
            "x", {"a": Fixed(1_000.0), "b": Fixed(2_500.0)}, Fixed(10_000.0),
            Fixed(0.5), 5,
        )
        result = run(spec, samples=50)
        self.assertTrue(np.all(result.annual_cost == 3_500.0))


class TestProbabilities(unittest.TestCase):
    def test_zero_risk_reduction_never_pays_back(self):
        result = run(fixed_spec(rr=0.0), samples=200)
        self.assertTrue(np.all(np.isinf(result.payback_years)))
        self.assertEqual(result.probability_never_pays_back(), 1.0)
        self.assertEqual(result.probability_payback_within(100.0), 0.0)

    def test_a_certain_case_has_probability_one(self):
        result = run(fixed_spec(), samples=200)  # payback 1.59 years
        self.assertEqual(result.probability_payback_within(2.0), 1.0)
        self.assertEqual(result.probability_payback_within(1.0), 0.0)

    def test_probability_is_monotone_in_the_hurdle(self):
        result = run(illustrative_spec(HUMAN_CAPITAL, 150_000, 0.8), samples=5_000)
        probs = [result.probability_payback_within(y) for y in (1.0, 1.5, 2.0, 3.0)]
        self.assertEqual(probs, sorted(probs))

    def test_probabilities_are_in_the_unit_interval(self):
        result = run(illustrative_spec(HUMAN_CAPITAL, 150_000, 0.8), samples=2_000)
        for value in (
            result.probability_payback_within(2.0),
            result.probability_never_pays_back(),
            result.probability_roi_at_least(1.0),
        ):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_the_broader_accounting_pays_back_sooner_than_the_narrow_one(self):
        """A structural property of the model, not an empirical finding."""
        narrow = run(illustrative_spec(TRADITIONAL, 150_000, 0.8), samples=5_000)
        broad = run(illustrative_spec(HUMAN_CAPITAL, 150_000, 0.8), samples=5_000)
        self.assertLess(
            float(np.median(broad.payback_years)),
            float(np.median(narrow.payback_years)),
        )

    def test_non_positive_hurdle_raises(self):
        result = run(fixed_spec(), samples=20)
        with self.assertRaises(ValueError):
            result.probability_payback_within(0.0)


class TestPercentilesAndSummary(unittest.TestCase):
    def test_percentiles_are_ordered(self):
        result = run(illustrative_spec(HUMAN_CAPITAL, 150_000, 0.8), samples=5_000)
        pct = result.percentiles("payback_years")
        self.assertLess(pct[5], pct[50])
        self.assertLess(pct[50], pct[95])

    def test_percentiles_reject_an_unknown_metric(self):
        result = run(fixed_spec(), samples=20)
        with self.assertRaises(ValueError):
            result.percentiles("irr")

    def test_percentiles_raise_when_nothing_is_finite(self):
        result = run(fixed_spec(rr=0.0), samples=20)
        with self.assertRaises(ValueError):
            result.percentiles("payback_years")

    def test_summary_contains_the_headline_keys(self):
        result = run(illustrative_spec(HUMAN_CAPITAL, 150_000, 0.8), samples=1_000)
        summary = result.summary(hurdle_years=3.0)
        for key in (
            "p_payback_within_hurdle", "p_never_pays_back",
            "payback_p5", "payback_median", "payback_p95", "median_roi",
        ):
            self.assertIn(key, summary)

    def test_format_result_mentions_the_seed(self):
        result = run(fixed_spec(), samples=50, seed=99)
        self.assertIn("99", format_result(result))


class TestSpecValidation(unittest.TestCase):
    def test_empty_components_raise(self):
        with self.assertRaises(ValueError):
            MonteCarloSpec("x", {}, Fixed(1.0), Fixed(0.5), 5)

    def test_non_positive_horizon_raises(self):
        with self.assertRaises(ValueError):
            MonteCarloSpec("x", {"a": Fixed(1.0)}, Fixed(1.0), Fixed(0.5), 0)

    def test_non_positive_samples_raise(self):
        with self.assertRaises(ValueError):
            run(fixed_spec(), samples=0)

    def test_non_positive_capex_draw_raises(self):
        spec = MonteCarloSpec("x", {"a": Fixed(1.0)}, Fixed(0.0), Fixed(0.5), 5)
        with self.assertRaises(ValueError):
            run(spec, samples=10)

    def test_out_of_range_risk_reduction_draw_raises(self):
        spec = MonteCarloSpec(
            "x", {"a": Fixed(1.0)}, Fixed(100.0), Uniform(0.5, 1.5), 5
        )
        with self.assertRaises(ValueError):
            run(spec, samples=200)

    def test_illustrative_spec_rejects_a_boundary_risk_reduction(self):
        with self.assertRaises(ValueError):
            illustrative_spec(HUMAN_CAPITAL, 150_000, 1.0)
        with self.assertRaises(ValueError):
            illustrative_spec(HUMAN_CAPITAL, 150_000, 0.0)

    def test_illustrative_spec_rejects_negative_spreads(self):
        with self.assertRaises(ValueError):
            illustrative_spec(HUMAN_CAPITAL, 150_000, 0.8, cost_spread=-0.1)


class TestIllustrativeSpec(unittest.TestCase):
    def test_it_covers_every_component(self):
        spec = illustrative_spec(HUMAN_CAPITAL, 150_000, 0.8)
        self.assertEqual(set(spec.components), set(HUMAN_CAPITAL.components))

    def test_capex_is_skewed_upward(self):
        """Capital projects overrun more often than they underrun."""
        spec = illustrative_spec(HUMAN_CAPITAL, 150_000, 0.8, capex_spread=0.15)
        self.assertGreater(spec.capex.mean_estimate, 150_000)

    def test_wider_spreads_widen_the_output(self):
        narrow = run(
            illustrative_spec(HUMAN_CAPITAL, 150_000, 0.8, cost_spread=0.05),
            samples=5_000,
        )
        wide = run(
            illustrative_spec(HUMAN_CAPITAL, 150_000, 0.8, cost_spread=0.60),
            samples=5_000,
        )
        self.assertLess(
            float(np.std(narrow.annual_cost)), float(np.std(wide.annual_cost))
        )

    def test_baseline_result_is_available_for_reference(self):
        spec = illustrative_spec(HUMAN_CAPITAL, 150_000, 0.8)
        self.assertIsNotNone(spec.baseline_result().payback_years)

    def _point_payback(self):
        return evaluate(
            HUMAN_CAPITAL, LOADING_STATION_CAPEX, LOADING_STATION_RISK_REDUCTION, 5
        ).payback_years

    def _symmetric_spec(self, cost_spread, capex_spread):
        """Inputs centred on their point values, with symmetric bands.

        Deliberately not illustrative_spec(), whose capex band is skewed
        upward on purpose. Symmetry is what makes the convexity claim below
        a statement about the arithmetic rather than about a chosen band.
        """
        capex = LOADING_STATION_CAPEX
        return MonteCarloSpec(
            label="symmetric",
            components={
                name: PERT(v * (1 - cost_spread), v, v * (1 + cost_spread))
                for name, v in HUMAN_CAPITAL.components.items()
            },
            capex=(
                PERT(capex * (1 - capex_spread), capex, capex * (1 + capex_spread))
                if capex_spread
                else Fixed(capex)
            ),
            risk_reduction=Fixed(LOADING_STATION_RISK_REDUCTION),
            horizon_years=5,
        )

    def test_mean_payback_exceeds_the_point_estimate_under_symmetric_inputs(self):
        """Payback divides by the uncertain quantity, so its MEAN is pulled up.

        payback = capex / (annual_cost x risk_reduction) is convex in the
        denominator, so by Jensen's inequality symmetric spread on the cost
        lines raises the mean above the point estimate. This is a property of
        the arithmetic, not of any chosen band.
        """
        point = self._point_payback()
        result = run(self._symmetric_spec(0.30, 0.0), samples=50_000, seed=7)
        self.assertGreater(float(np.mean(result.payback_years)), point)

    def test_symmetric_capex_spread_alone_does_not_shift_the_mean(self):
        """Payback is LINEAR in capex, so symmetric capex noise cancels.

        Complements the test above: it localises the mean shift to the
        denominator, and rules out "any spread raises the mean".
        """
        point = self._point_payback()
        result = run(self._symmetric_spec(0.0, 0.15), samples=50_000, seed=7)
        self.assertAlmostEqual(
            float(np.mean(result.payback_years)), point, delta=0.005
        )

    def test_default_spec_median_shift_comes_from_the_capex_skew(self):
        """Guards against reading illustrative_spec's median as structural.

        illustrative_spec() skews capex upward on purpose, which lifts the
        median. Under a symmetric band the median sits essentially at the
        point estimate, so the median shift is an artefact of a chosen
        placeholder and must not be reported as a finding.
        """
        point = self._point_payback()
        skewed = run(
            illustrative_spec(
                HUMAN_CAPITAL, LOADING_STATION_CAPEX, LOADING_STATION_RISK_REDUCTION
            ),
            samples=50_000,
            seed=7,
        )
        symmetric = run(self._symmetric_spec(0.30, 0.15), samples=50_000, seed=7)

        self.assertGreater(float(np.median(skewed.payback_years)), point)
        self.assertAlmostEqual(
            float(np.median(symmetric.payback_years)), point, delta=0.005
        )


class TestDefaults(unittest.TestCase):
    def test_default_sample_count_is_sane(self):
        self.assertGreaterEqual(DEFAULT_SAMPLES, 10_000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
