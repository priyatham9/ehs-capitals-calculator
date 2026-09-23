# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog and this project uses semantic versioning.

## [Unreleased]

### Added
- Calculator: pin a scenario and compare paybacks and ROI against the current inputs
- Calculator: copy a plain-text decision memo for a capital request (inputs, both accountings, verdict, scenario link)
- Calculator: copy-link and reset controls next to the outputs; the page now follows hash edits and back/forward
- Calculator: spring-animated payback figures, verdict flip cue, formatted capital-cost echo, inline range validation, Shift+Arrow steps by 10

### Fixed
- A malformed or truncated scenario link no longer puts NaN into the form

## [0.1.0] - 2026-09-09

### Added
- Cost-benefit calculator (arithmetic model, not statistics) computing EHS capital investment returns two ways: traditional direct costs and human-capital-inclusive accounting
- Simple four-line model: annual benefit, residual cost, payback, ROI; transparency on all assumptions and limitations
- Python library (capitals.py), CLI, and pure-standard-library subcommands (example, compare, tornado) plus optional numpy Monte Carlo simulation
- Deterministic tornado sensitivity analysis (one-at-a-time) ranking inputs by impact on payback and breakeven calculation by bisection
- Probabilistic Monte Carlo module with distributions (Triangular, PERT, Uniform, Normal, BoundedBeta, Fixed) and seeded reproducibility
- Browser calculator at docs/index.html (no build, no dependencies, no network) pinned to Python implementation by parity test
