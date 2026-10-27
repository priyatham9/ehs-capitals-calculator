# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog and this project uses semantic versioning.

## [Unreleased]

### Added
- `uncertainty` subcommand and `src/uncertainty.py` (standard library only): exact interval bounds on payback, ROI and net ROI over assumed input ranges (`--spread`, `--range NAME=LOW:HIGH`) plus seeded uniform sampling for percentiles and P(payback within hurdle); ranges are labelled as assumptions; tests check bounds against brute-force box corners and seeded reproducibility
- `.zenodo.json` and `codemeta.json` (title, description, creators from `CITATION.cff`, MIT licence, keywords, related identifiers for the sibling repositories and the programme site; no DOI or ORCID) so a release is archived with correct metadata once Zenodo is linked; `CITATION.cff` gains `keywords`. `tools/check_citation.py` (stdlib, run by `.github/workflows/citation.yml`) checks the required CFF 1.2.0 fields and that `CITATION.cff`, `.zenodo.json`, `codemeta.json` and `LICENSE` agree on title, authors, keywords and licence. No release or tag created.
- `heat` subcommand and `src/heat.py`: NWS heat index (Rothfusz regression with adjustments) and status against the initial (80 F) and high (90 F) heat-index triggers in OSHA's proposed heat rule (89 FR 70698), with tests
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
