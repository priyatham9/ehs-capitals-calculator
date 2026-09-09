# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog and this project uses semantic versioning.

## [0.1.0] - 2026-09-09

### Added
- Cost-benefit calculator (arithmetic model, not statistics) computing EHS capital investment returns two ways: traditional direct costs and human-capital-inclusive accounting
- Simple four-line model: annual benefit, residual cost, payback, ROI; transparency on all assumptions and limitations
- Python library (capitals.py), CLI, and pure-standard-library subcommands (example, compare, tornado) plus optional numpy Monte Carlo simulation
- Deterministic tornado sensitivity analysis (one-at-a-time) ranking inputs by impact on payback and breakeven calculation by bisection
- Probabilistic Monte Carlo module with distributions (Triangular, PERT, Uniform, Normal, BoundedBeta, Fixed) and seeded reproducibility
- Browser calculator at docs/index.html (no build, no dependencies, no network) pinned to Python implementation by parity test
