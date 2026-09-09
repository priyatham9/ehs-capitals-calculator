# Contributing

## Running Tests

```bash
cd /path/to/ehs-capitals-calculator
python3 -m unittest discover -s tests -v
```

166 tests, standard library only for the core model and tests.

## Dependency Policy

This project uses Python 3.9 or later with:
- Standard library only for capitals.py, sensitivity.py, and cli.py (subcommands: example, compare, tornado)
- numpy (required only for the montecarlo subcommand; optional, with graceful degradation if not installed)
- pandas (for test fixtures and examples, not required at runtime)

## Numbers in the Repository

Every number in README.md and documentation comes from one of:

1. The published source article (Brandon 2025, LeadingEHS.com), transcribed and committed
2. A committed script that computes derived quantities (net ROI, tornado rankings, breakeven points)
3. Illustrative placeholders in the Monte Carlo example, clearly labelled as non-findings

Specific rules:

- The worked example cost lines ($55k, $118k, etc.) are transcribed from the source article and pinned in `tests/TestPublishedExample`
- Breakeven values are verified against closed-form arithmetic in tests
- The JavaScript and Python implementations are linked by `tests/parity_cases.json`, which pins six test cases with values produced by the browser JavaScript
- Monte Carlo placeholders in `illustrative_spec()` are test fixtures only, not measured data

## Synthetic Data

This repository contains illustrative placeholders, not synthetic data files. The Monte Carlo module in `montecarlo.py` includes `illustrative_spec()` which demonstrates the simulation machinery with placeholder distributions and parameters.

- These placeholders are designed for demonstration and testing only
- They must never be reported as findings or measurements
- The test suite enforces that the median stays on the point estimate under symmetric distributions (a property of the arithmetic, not of the data)
