# ehs-capitals-calculator

[![tests](https://github.com/priyatham9/ehs-capitals-calculator/actions/workflows/tests.yml/badge.svg)](https://github.com/priyatham9/ehs-capitals-calculator/actions/workflows/tests.yml)

A cost-benefit calculator for EHS capital investment that computes the same
investment two ways: counting only directly booked costs, and additionally
counting human-capital costs that an incident causes but that rarely get
charged against it.

The model is arithmetic, not statistics. It is small enough to read in full,
and this README states plainly what it assumes and where it breaks.

- Python library, CLI, tornado sensitivity, and Monte Carlo modules
- A single-file browser calculator at `docs/index.html` (no build step, no
  network, no dependencies)
- 166 tests, including a fixture pinning the JavaScript and Python
  implementations to the same numbers

## What problem this addresses

A safety control competes for capital against projects that can quote a return.
The costs it avoids are mostly counted in one narrow way: repair bills,
material loss, workers' compensation. Costs that are real but land in other
budgets - the shift that ran short-handed, the replacement operator who took
four months to reach full rate, the crew that got slower and more cautious for
a quarter - do not appear on the incident's ledger, so they do not appear in
the business case either.

The calculator makes the difference explicit rather than rhetorical. Enter the
same investment twice and see what changes.

**This is an accounting-scope argument, not a discovery.** The tool cannot tell
you whether the human-capital numbers you enter are right. It can only show you
what follows if they are, and how hard you would have to lean on them to change
the answer.

## The model

Four lines, and that is the whole thing:

```
annual_benefit = annual_incident_cost x risk_reduction
residual_cost  = annual_incident_cost - annual_benefit
payback_years  = capex / annual_benefit
roi            = (annual_benefit x horizon_years) / capex
```

The two accountings differ only in which cost lines are summed into
`annual_incident_cost`. The human-capital accounting is a strict superset of
the traditional one.

### Worked example

Reproduces the loading-station comparison published in Brandon, C. (27 May
2025), ["You Can't Manage What You Don't Value: Reimagining Capital in
Organizations"][source], LeadingEHS.com.

| Cost line | Annual | Counted by |
|---|---:|---|
| Equipment downtime | $30,000 | both |
| Material loss / spills | $10,000 | both |
| Injury costs | $15,000 | both |
| Lost work time | $20,000 | human-capital only |
| Retraining due to turnover | $8,000 | human-capital only |
| Productivity loss | $25,000 | human-capital only |
| Morale / team performance | $10,000 | human-capital only |

At $150,000 capital cost and an assumed 80% risk reduction:

| | Traditional | Human-capital-inclusive |
|---|---:|---:|
| Annual incident cost | $55,000 | $118,000 |
| Residual cost (20%) | $11,000 | $23,600 |
| Annual saving | $44,000 | $94,400 |
| Payback | 3.41 yr | 1.59 yr |
| 5-year ROI (cumulative) | 147% | 315% |
| 5-year net ROI | 47% | 215% |

Every figure above except net ROI appears in the source article's own table,
including the intermediate residual-cost and annual-saving rows. Net ROI is
this repository's addition; see the ROI caveat below.

## Install and run

Python 3.9 or later. `numpy` is the only third-party dependency, and only the
Monte Carlo module needs it. `capitals.py`, `sensitivity.py` and `cli.py` are
pure standard library: the `example`, `compare` and `tornado` subcommands run
on a bare interpreter, and `montecarlo` is imported only when that subcommand
is invoked. Without numpy installed, `python3 src/cli.py montecarlo` exits 1
with an install hint and nothing else is affected.

```bash
git clone https://github.com/priyatham9/ehs-capitals-calculator
cd ehs-capitals-calculator
python3 -m unittest discover -s tests -v
```

### Command line

```bash
python3 src/cli.py example                     # the published comparison
python3 src/cli.py compare --capex 250000 --risk-reduction 0.6 --hurdle 3
python3 src/cli.py tornado --hurdle 2          # what moves the answer
python3 src/cli.py montecarlo --samples 100000 # payback probability
```

Supply your own cost lines with repeated `name=value` flags. `--cost` adds a
line both accountings count; `--human-cost` adds one only the second counts.

```bash
python3 src/cli.py compare \
 --capex 400000 --risk-reduction 0.55 --horizon 7 --hurdle 4 \
 --cost downtime=90000 --cost workers_comp=45000 \
 --human-cost lost_time=60000 --human-cost retraining=25000
```

Every subcommand takes `--json`. Every subcommand involving randomness takes
`--seed`, fixed by default.

### Library

```python
import sys; sys.path.insert(0, "src")
from capitals import CostModel, evaluate

model = CostModel("plant A", {"downtime": 90_000.0, "lost_time": 60_000.0})
result = evaluate(model, capex=400_000, risk_reduction=0.55, horizon_years=7)
print(result.payback_years, result.roi, result.net_roi)
```

### Browser

Open `docs/index.html` directly, or serve the folder. It has no dependencies
and makes no network requests. It starts loaded with the published worked
example and can be reset to it at any time with the button at the top left.

## What the sensitivity tools are for

The point estimate is the least interesting output. A payback of "1.59 years"
is the arithmetic consequence of nine numbers, one of which - risk reduction - 
is a guess about the future. Both sensitivity tools exist to keep that visible.

### `sensitivity.py` - deterministic

`tornado()` swings each input across a range one at a time and ranks inputs by
how far the answer moves. On the published example at +/-30%, the order is:

```
risk_reduction   1.00 yr
capex            0.95 yr
equipment_downtime  0.24 yr
...
horizon_years    0.00 yr
```

Risk reduction - the one input nobody can measure in advance - moves the answer
further than every cost line combined. That is the finding worth acting on.

`decision_sensitivity()` asks the sharper question: which inputs can, on their
own, move the answer across a decision threshold? A wide swing does not matter
if the whole range clears the hurdle. Breakeven points are found by bisection
and agree with the closed forms, which the tests assert directly:

- capex breakeven = `hurdle x annual_benefit`
- risk-reduction breakeven = `capex / (hurdle x annual_cost)`

### `montecarlo.py` - probabilistic

Replaces each point input with a distribution (`Triangular`, `PERT`, `Uniform`,
truncated `Normal`, `BoundedBeta`, `Fixed`) and reports the distribution of the
answer, including the probability of paying back within N years.

Seeded via `numpy.random.default_rng` with a fixed default, so a run is
reproducible from its seed and sample count alone.

One structural result worth knowing, stated carefully. Payback divides by the
uncertain quantity - `capex / (annual_cost x risk_reduction)` - so it is convex
in its denominator. By Jensen's inequality, symmetric uncertainty on the cost
lines therefore pulls the **mean** payback above the point estimate. Payback is
linear in capex, so symmetric capex noise does not move the mean at all. Both
halves of that are pinned by tests.

The **median** is a different matter, and it is easy to over-read. Under
symmetric bands the simulated median sits essentially *on* the point estimate,
not above it. The median does come out above the point estimate when you run
`illustrative_spec()`, but only because that spec deliberately skews capex
upward; that is a consequence of a placeholder someone chose, not a property of
the arithmetic, and it must not be reported as a finding. A test asserts the
distinction so the two cannot be conflated.

## Limitations

Read this section before using any number this produces in a funding request.

**ROI here is cumulative benefit over investment, not net return.** A reported
147% means accumulated benefits reached 1.47x the capital cost over the
horizon, not that 147% was cleared as profit. This is the convention the source
article uses, and it is the more flattering of the two. `net_roi` is reported
alongside it and is always exactly 100 percentage points lower. If a finance
reviewer expects "ROI" to mean net return, hand them `net_roi`.

**Risk reduction is an input you supply, not something the tool knows.** This
is the largest weakness in the whole model and no amount of downstream
machinery repairs it. The tool does not know your control, your process, or
your failure modes. It multiplies by whatever you type. The tornado and
breakeven views exist specifically so that a reader can see how much of the
conclusion is resting on that one assumption - usually most of it.

**Nothing is discounted.** A dollar saved in year 5 counts the same as one
saved in year 1. At any realistic cost of capital this overstates long-horizon
ROI, and the error grows with the horizon. Payback in years is unaffected;
the ROI figures are not. There is no NPV or IRR here, deliberately - adding
them would imply a rigour the rest of the model does not have.

**Benefits are assumed flat and permanent.** One annual cost and one risk
reduction, unchanged across the horizon. No control degradation, no maintenance
or training cost, no residual value, no production growth.

**Counting more cost can never weaken the case, by construction.** Every added
line is non-negative, so the human-capital accounting always shows payback at
least as fast. That is a property of the arithmetic, not evidence about the
world. It means the model cannot ever falsify the argument it is built to make,
and it is a reason to treat the direction of the result as uninformative and
scrutinise the magnitudes instead.

**The human-capital cost lines are estimates, not invoices.** Lost work time,
retraining, productivity and morale are real costs, but attributing a dollar
figure to them is a judgement call, and different reasonable analysts will
differ by a lot. The second accounting inherits all of that uncertainty. In the
published example, $63,000 of the $118,000 annual cost - over half - is in
lines nobody receives a bill for.

**Monte Carlo propagates uncertainty; it does not reduce it.** Feeding the
model a distribution centred on a wrong number produces a confident
distribution centred on a wrong answer, and the spread makes it look
considered. The output describes your stated beliefs, not the world.

**Monte Carlo inputs are sampled independently.** In reality the cost lines are
driven by a shared underlying incident rate, so a bad year is bad across
several lines at once. Independent sampling lets those errors cancel, which
understates the spread of total annual cost. Treat the reported intervals as
narrower than the truth. Correlated sampling is not implemented.

**One-at-a-time sensitivity ignores interactions.** The tornado holds every
other input at baseline, so it understates joint risk for the same reason.
Where inputs move together, use the Monte Carlo module.

**Tornado rankings depend on the ranges you choose.** Bars built from a uniform
+/-30% band rank inputs by the width of a band someone picked, not by real
uncertainty. `relative_ranges()` is a convenience for exploration; supply
ranges you can defend before quoting a ranking.

**No empirical data is used anywhere in this repository.** There is no dataset,
no fitted parameter, and no statistical estimate. Every number is either
supplied by the user, or transcribed from the source article's published table
and marked as such. The Monte Carlo spreads in `illustrative_spec()` are
labelled placeholders chosen to demonstrate the machinery - they are not
measured, not elicited, and must not be reported as findings.

## Provenance of the numbers

| Number | Where it comes from |
|---|---|
| All seven cost lines, $55,000, $118,000, $11,000, $23,600, $44,000, $94,400, 3.41 yr, 1.59 yr, 147%, 315%, $150,000 capex, 80% risk reduction | Transcribed from the cost table in [Brandon (2025)][source]. Retrieved and checked against this implementation on 3 September 2026. |
| Net ROI (47%, 215%) | Computed here; not in the source. |
| Tornado orderings and breakeven values | Computed from the above by this repository's code; verified against closed-form arithmetic in the tests. |
| Monte Carlo spreads | Illustrative placeholders. Not measured. See `illustrative_spec()`. |

An earlier revision of `src/capitals.py` split the two published totals into
invented line items with a comment stating the article did not publish a
breakdown. The article does publish one. The line items were replaced with the
transcribed values; the totals, and therefore every test, were unaffected.

## Repository layout

```
src/capitals.py       the model, the worked example, input validation
src/sensitivity.py    tornado, breakeven search, two-way grids
src/montecarlo.py     distributions, seeded simulation, probabilities
src/cli.py            argparse front end for all of the above
docs/index.html       self-contained browser calculator
tests/                166 tests
tests/parity_cases.json  pins the JavaScript and Python implementations together
```

### Keeping two implementations honest

`docs/index.html` re-implements the model in JavaScript so the page runs with
no build step and no server. Two implementations of one formula drift apart
silently, so `tests/parity_cases.json` holds six cases - including awkward
decimals and the zero-benefit edge case - with values produced by the
JavaScript running in a browser. `TestJavaScriptParity` asserts Python
reproduces them. All six agreed to full double precision when the fixture was
written. If you change the arithmetic in one implementation, regenerate the
fixture and re-run the tests.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

166 tests. The suite pins the published figures, asserts the closed-form
breakevens rather than recorded outputs, checks Monte Carlo reproducibility
under a fixed seed, and confirms that the vectorised simulation agrees exactly
with the scalar model when all distributions are degenerate.

## Related work

The author's [ehs-benchmarks](https://github.com/priyatham9/ehs-benchmarks)
processes 1.18M public OSHA establishment filings into injury-rate benchmarks.
It is separate work with a different basis: that repository analyses real
regulatory data, whereas this one is a transparent calculator over inputs a
user supplies. Findings from the former do not transfer to, or support, any
output of the latter.

## License

MIT. See [LICENSE](LICENSE).

Security concerns can be reported in [SECURITY.md](SECURITY.md).

The worked example is attributed to its author and linked above; the arithmetic
implementing it is original to this repository.

[source]: https://leadingehs.com/2025/05/27/you-cant-manage-what-you-dont-value-reimagining-capital-in-organizations/
