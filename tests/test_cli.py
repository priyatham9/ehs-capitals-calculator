"""Tests for the command-line interface.

The CLI is the surface most likely to be used by someone who has not read the
source, so the tests concentrate on two things: that the published example
comes out of `example` unaltered, and that bad input is rejected with a clear
error rather than silently producing a plausible wrong number.
"""

import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cli  # noqa: E402


def capture(argv):
    """Run the CLI and return (exit_code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class TestExample(unittest.TestCase):
    def test_example_runs_cleanly(self):
        code, out, _ = capture(["example"])
        self.assertEqual(code, 0)
        self.assertIn("Traditional", out)

    def test_example_prints_the_published_figures(self):
        _, out, _ = capture(["example"])
        for figure in ("55,000", "118,000", "3.41", "1.59", "147%", "315%"):
            self.assertIn(figure, out)

    def test_example_cites_its_source(self):
        _, out, _ = capture(["example"])
        self.assertIn("leadingehs.com", out)

    def test_example_json_carries_the_published_figures(self):
        _, out, _ = capture(["example", "--json"])
        payload = json.loads(out)
        traditional, enhanced = payload["results"]
        self.assertEqual(traditional["annual_cost"], 55_000)
        self.assertEqual(enhanced["annual_cost"], 118_000)
        self.assertAlmostEqual(enhanced["payback_years"], 1.59, places=2)

    def test_example_lists_every_cost_line(self):
        _, out, _ = capture(["example"])
        for line in ("equipment_downtime", "productivity_loss", "morale_team_performance"):
            self.assertIn(line, out)


class TestCompare(unittest.TestCase):
    def test_defaults_reproduce_the_worked_example(self):
        _, out, _ = capture(["compare"])
        self.assertIn("3.41", out)
        self.assertIn("1.59", out)

    def test_custom_costs_are_used(self):
        _, out, _ = capture(
            ["compare", "--capex", "100000", "--risk-reduction", "0.5",
             "--cost", "downtime=40000", "--human-cost", "morale=60000", "--json"]
        )
        traditional, enhanced = json.loads(out)["results"]
        self.assertEqual(traditional["annual_cost"], 40_000)
        self.assertEqual(enhanced["annual_cost"], 100_000)
        self.assertAlmostEqual(enhanced["payback_years"], 2.0)

    def test_hurdle_verdicts_are_reported(self):
        _, out, _ = capture(["compare", "--hurdle", "2"])
        self.assertIn("passes", out)
        self.assertIn("fails", out)

    def test_the_assumption_warning_is_always_printed(self):
        _, out, _ = capture(["compare"])
        self.assertIn("assumption you supplied", out)

    def test_dollar_signs_and_commas_are_accepted(self):
        _, out, _ = capture(["compare", "--cost", "downtime=$40,000", "--json"])
        self.assertEqual(json.loads(out)["results"][0]["annual_cost"], 40_000)

    def test_zero_risk_reduction_reports_never(self):
        _, out, _ = capture(["compare", "--risk-reduction", "0"])
        self.assertIn("never", out)

    def test_json_is_valid_and_complete(self):
        _, out, _ = capture(["compare", "--json"])
        for result in json.loads(out)["results"]:
            for key in ("annual_cost", "residual_cost", "payback_years", "roi", "net_roi"):
                self.assertIn(key, result)


class TestTornado(unittest.TestCase):
    def test_runs_and_ranks_risk_reduction_first(self):
        code, out, _ = capture(["tornado"])
        self.assertEqual(code, 0)
        self.assertIn("risk_reduction", out)

    def test_reports_the_flip_points(self):
        _, out, _ = capture(["tornado", "--hurdle", "2"])
        self.assertIn("188,800", out)  # closed form: 2 x 94,400

    def test_roi_metric_is_supported(self):
        _, out, _ = capture(["tornado", "--metric", "roi"])
        self.assertIn("roi", out)

    def test_traditional_model_can_be_selected(self):
        _, out, _ = capture(["tornado", "--model", "traditional", "--json"])
        self.assertEqual(json.loads(out)["model"], "Traditional")

    def test_json_bars_are_sorted_by_swing(self):
        _, out, _ = capture(["tornado", "--json"])
        swings = [b["swing"] for b in json.loads(out)["bars"] if b["swing"] is not None]
        self.assertEqual(swings, sorted(swings, reverse=True))

    def test_reports_when_nothing_can_flip_the_verdict(self):
        _, out, _ = capture(["tornado", "--hurdle", "20", "--spread", "0.05"])
        self.assertIn("No single input", out)


class TestMonteCarlo(unittest.TestCase):
    def test_runs_and_reports_a_probability(self):
        code, out, _ = capture(["montecarlo", "--samples", "2000"])
        self.assertEqual(code, 0)
        self.assertIn("P(payback within", out)

    def test_is_reproducible_across_invocations(self):
        _, first, _ = capture(["montecarlo", "--samples", "3000", "--seed", "7"])
        _, second, _ = capture(["montecarlo", "--samples", "3000", "--seed", "7"])
        self.assertEqual(first, second)

    def test_different_seeds_change_the_output(self):
        _, a, _ = capture(["montecarlo", "--samples", "3000", "--seed", "1"])
        _, b, _ = capture(["montecarlo", "--samples", "3000", "--seed", "2"])
        self.assertNotEqual(a, b)

    def test_flags_its_spreads_as_illustrative(self):
        _, out, _ = capture(["montecarlo", "--samples", "500"])
        self.assertIn("illustrative", out)

    def test_json_flags_its_spreads_as_illustrative(self):
        _, out, _ = capture(["montecarlo", "--samples", "500", "--json"])
        self.assertTrue(json.loads(out)["spreads_are_illustrative"])

    def test_json_probability_is_a_proportion(self):
        _, out, _ = capture(["montecarlo", "--samples", "2000", "--json"])
        for result in json.loads(out)["results"]:
            self.assertGreaterEqual(result["p_payback_within_hurdle"], 0.0)
            self.assertLessEqual(result["p_payback_within_hurdle"], 1.0)


class TestArgumentValidation(unittest.TestCase):
    """Bad input must fail loudly. argparse exits with code 2."""

    def assert_exits(self, argv):
        err = io.StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                cli.main(argv)
        self.assertNotEqual(ctx.exception.code, 0)
        return err.getvalue()

    def test_percentage_instead_of_proportion_is_rejected(self):
        """The likeliest user error: typing 80 rather than 0.80."""
        message = self.assert_exits(["compare", "--risk-reduction", "80"])
        self.assertIn("0.8", message)

    def test_negative_capex_is_rejected(self):
        self.assert_exits(["compare", "--capex", "-5"])

    def test_zero_capex_is_rejected(self):
        self.assert_exits(["compare", "--capex", "0"])

    def test_zero_horizon_is_rejected(self):
        self.assert_exits(["compare", "--horizon", "0"])

    def test_malformed_cost_pair_is_rejected(self):
        self.assert_exits(["compare", "--cost", "downtime"])

    def test_non_numeric_cost_is_rejected(self):
        self.assert_exits(["compare", "--cost", "downtime=lots"])

    def test_negative_cost_is_rejected(self):
        self.assert_exits(["compare", "--cost", "downtime=-100"])

    def test_unknown_metric_is_rejected(self):
        self.assert_exits(["tornado", "--metric", "irr"])

    def test_missing_subcommand_is_rejected(self):
        self.assert_exits([])

    def test_a_component_in_both_groups_is_rejected(self):
        message = self.assert_exits(
            ["compare", "--cost", "x=1000", "--human-cost", "x=2000"]
        )
        self.assertIn("exactly one accounting", message)


class TestHelpers(unittest.TestCase):
    def test_parse_cost_pairs_handles_an_empty_input(self):
        self.assertEqual(cli.parse_cost_pairs(None), {})
        self.assertEqual(cli.parse_cost_pairs([]), {})

    def test_parse_cost_pairs_strips_formatting(self):
        parsed = cli.parse_cost_pairs(["a = $1,250.50 "])
        self.assertAlmostEqual(parsed["a"], 1250.50)

    def test_human_formats_large_and_small_values_readably(self):
        self.assertEqual(cli._human(188_800.0), "188,800")
        self.assertEqual(cli._human(0.6356), "0.6356")

    def test_jsonable_maps_infinity_to_none(self):
        self.assertIsNone(cli._jsonable(float("inf")))
        self.assertEqual(cli._jsonable(2.5), 2.5)

    def test_parser_builds_without_error(self):
        self.assertIsNotNone(cli.build_parser())


class TestMonteCarloDefaultsMatch(unittest.TestCase):
    """cli.py mirrors two montecarlo constants to keep numpy out of import.

    The mirroring is what lets `example`, `compare` and `tornado` run without
    numpy. These tests make the duplication safe: if montecarlo changes a
    default, the copy in cli.py must change with it.
    """

    def test_seed_default_matches_montecarlo(self):
        import montecarlo

        self.assertEqual(cli.DEFAULT_SEED, montecarlo.DEFAULT_SEED)

    def test_samples_default_matches_montecarlo(self):
        import montecarlo

        self.assertEqual(cli.DEFAULT_SAMPLES, montecarlo.DEFAULT_SAMPLES)

    def test_cli_does_not_import_montecarlo_at_module_level(self):
        """The non-Monte-Carlo subcommands must not need numpy.

        Asserted on the source text rather than by manipulating sys.modules,
        because montecarlo is legitimately imported by other tests in the run.
        """
        source = (
            Path(cli.__file__).resolve().read_text()
        )
        module_level = source.split("def ", 1)[0]
        self.assertNotIn("from montecarlo import", module_level)
        self.assertNotIn("import montecarlo", module_level)


if __name__ == "__main__":
    unittest.main(verbosity=2)
