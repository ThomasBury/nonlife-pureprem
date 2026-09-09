# Remediation handoff

Updated: 2026-09-07.

The completed phase scope and acceptance checks are in [PLAN.md](PLAN.md). Read that file and
[AGENTS.md](AGENTS.md) before continuing after a context reset.

## Current state

| Phase | Status | Next action |
| --- | --- | --- |
| 1: Tweedie likelihood and probabilities | Implemented and verified; stopped for review | Preserve this work and its regression tests |
| 2: Conditional distributions and plot repairs | Implemented and verified | Preserve this work and its regression tests |
| 3: Interpretation and fresh book execution | Implemented and verified | Preserve this work |
| Uncapped exploration before fitting | Implemented and verified | Stop for review |

The latest user request authorized one additional phase: uncapped training
exploration before power selection or model fitting. Earlier phases are complete.
The new phase is implemented and verified; stop here for review. Nothing
has been committed or published by this agent.

**Backward compatibility is not required.** Remove obsolete interfaces and
migrate repository callers directly. Do not retain aliases, wrappers,
optional legacy arguments, or fallback paths to avoid breaking old behavior.
This project instruction is also recorded in AGENTS.md.

## Phase 1 implemented

Changes are in the working tree, not committed by this agent:

- src/nonlife_pureprem/pure_premium.py exports
  tweedie_log_likelihood(y, mean, exposure, tweedie_power, dispersion) and
  fit_tweedie_dispersion(y, mean, exposure, tweedie_power).
- Likelihood uses \(a_i=e_i^{1/(2-p)}\), GLUM scalar dispersion with unit
  likelihood weights, and the positive-observation Jacobian only. Inputs
  and transformed values are validated.
- Dispersion maximizes that likelihood over log dispersion with means fixed.
  The exposure-adjusted Pearson estimate is only the starting value.
  Optimization failure or nonfinite results raise; no moment substitution.
- select_tweedie_power retains exposure-weighted mean fitting, candidate
  order, profile columns, and its return contract. It fits dispersion on
  development training and uses that same value for validation scoring.
- tweedie_cdf_series and tweedie_sf_series share a private calculation.
  It uses Poisson log-PMF weights, direct Gamma CDF/SF terms, endpoint
  handling, and a relative omitted-mass stopping bound. Means and
  dispersions broadcast; defaults are tol=1e-10, max_terms=10_000.
  Exhausted convergence raises.
- Package exports, the power-selection explanation, and obsolete CMP prose
  and dead imports were updated. Count-tail smoke expects three implemented
  curves.
- Existing tests were extended with zero-retaining simulations, likelihood
  references, dispersion optima, power recovery, endpoints, broadcasting,
  large intensities, tiny tails, invalid inputs, and explicit failures.

## Verification already completed

Last full Phase 1 run on 2026-09-05:

~~~text
uv run python -m pytest tests/ -q
40 passed, 64 subtests passed in 45.60s

uv run ruff check .
All checks passed!

uv run ruff format --check .
5 files already formatted
~~~

The changed-file diff check and the repository-wide check excluding the
pre-existing PDF passed. The unrestricted git diff --check flags whitespace
in the pre-existing docs/compound-poisson-gamma-theory.pdf change.

Numerical evidence worth retaining:

- For \(p=1.5,\mu=1,\phi=0.1\), survival at 10 is approximately
  \(3.982997209043982\times10^{-43}\); at 20 it is approximately
  \(1.648721054305175\times10^{-107}\).
- Tiny-tail assertions use relative tolerance and zero absolute tolerance.
  An 80-digit standard-library decimal reference covers the smaller tail:
  SciPy's noncentral chi-square CDF returned zero for it in this environment.
- GLUM 3.4.1 density evaluation remains intentionally in use. In the
  independent log-likelihood regression cases, non-1.5 powers differ from
  direct mixture sums by a few millionths. The tests document their explicit
  tolerances; do not treat those as exact density equality.
- The power simulation retains zero claims and verifies that dispersion
  fitting receives development-training rows and aligned predictions.

## Phase 2 implemented

The resumed working tree already contained partial Phase 2 source and chapter
changes despite the old status table. This session reviewed that work,
completed its regression coverage, fixed the NB Poisson-limit calculation,
and finished validation. Changes remain uncommitted.

Final public interfaces (no legacy aliases):

~~~python
fit_gamma_dispersion(y, mean, claim_count)
fit_negative_binomial_shape(counts, mean)
poisson_cdf_diagnostic(data, mean, axes=None)
poisson_ccdf_diagnostic(data, mean, nb_shape, axes=None)
gamma_cdf_diagnostic(data, mean, dispersion, axes=None)
gamma_ccdf_diagnostic(data, mean, dispersion, axes=None)
tweedie_cdf_diagnostic(data, mean, tweedie_power, dispersion, axes=None)
tweedie_ccdf_diagnostic(data, mean, tweedie_power, dispersion, axes=None)
conditional_dispersion_table(data, mean, by, response, tweedie_power)
conditional_dispersion_diagnostic(
    data, predictions, by, tweedie_power, dispersions, axes=None
)
~~~

- Plotters require nonempty eligible rows and positive predicted rates in a
  pandas Series whose unique index matches the supplied rows in order.
  They neither fit nor sample. Gamma/Tweedie base dispersion and NB shape
  must be supplied explicitly.
- Count references use Poisson(exposure * predicted frequency). Positive
  policy-average severity uses Gamma shape count/phi and scale phi*mean/count.
  Capped totals use Tweedie mean exposure*rate and dispersion
  phi*exposure**(1-p). Empirical and reference curves use the same rows and
  exposure weights (counts/totals) or claim-count weights (severity).
- Weighted empirical curves aggregate identical outcomes, including the full
  zero atom. Conditional references evaluate 128 thresholds in batches of
  at most 8 thresholds by 512 policies; Tweedie terms accumulate sequentially.
- Gamma dispersion fits every eligible policy's average-severity log density
  once, without multiplying by count a second time. NB fits positive real
  shape against fixed predicted policy counts. Both validate inputs and
  optimizer outcomes; neither substitutes moments after a failed fit.
- NB log likelihood uses log beta and log1p to avoid cancellation near the
  Poisson limit. The regression initially exposed a false finite shape of
  about 32.8 million for counts=means=1; the corrected fitter returns infinity,
  and the SF overlay explicitly labels the Poisson limit.
- conditional_dispersion_table reports area, n, risk_volume, observed_rate,
  predicted_rate, and dispersion. Scaled residual squares are averaged by
  row count at fixed powers 1, 2, and selected Tweedie p. The diagnostic
  compares these with 1 or the fitted Gamma/Tweedie base dispersion.
- Deleted _fit_tweedie_mle, mean_variance_table, mean_variance_diagnostic,
  _INITIAL_VARIANCE_POWER, obsolete fit imports, and their legacy tests and
  walkthroughs. Package exports and every source caller are migrated.
- Both lift tables aggregate identical scores/ratios before cumulative
  interpolation and spread each tied block proportionally across equal
  exposure bins. They preserve weighted totals and average individual A/B
  ratios for prediction_ratio. Invalid, empty, and misaligned inputs raise.
- Driver-age bands remain ordered categories through grouping and sorting;
  plotting converts only tick labels to strings. Linear observed hexbins
  include zero and pad constant prediction ranges. Empty visible bins and
  empty positive-only panels safely share log colour normalization.

### Training and sampling flow

The chapter splits immediately after preparation. The full-source audit
contains structural counts only; outcome averages use training policies.
Power selection uses development rows, then final training GLUM fits precede
the conditional diagnostics. Every eligible training row for each component
is predicted and passed to its dispersion/shape fitter before any diagnostic
sampling. Sampling uses min(20_000, eligible rows), seed 42; indexed predictions
are selected in the sampled order. Each CDF/SF pair reuses that exact sample,
and titles report its size. Area dispersion uses all eligible training rows.
Uncapped individual claim tails are a separate descriptive view, filtered to
training policy IDs. Test policies are reserved for the final audit.

### Phase 2 validation

Final complete run on 2026-09-05:

~~~text
uv sync --locked --all-groups
Resolved 131 packages; audited 128 packages

uv run python -m pytest tests/ -q
40 passed, 132 subtests passed in 42.26s

uv run ruff check .
All checks passed!

uv run ruff format --check .
7 files already formatted

git diff --check -- . ':!docs/compound-poisson-gamma-theory.pdf'
Passed
~~~

Coverage includes independent heterogeneous Poisson/Gamma references and a
noncentral-chi-square Tweedie reference, empirical weights and alignment,
bounded batches, all-training-row fitting before deterministic sampling,
CDF/SF sample identity, uncapped training claim IDs, unequal-volume residual
dispersion, real NB shape below one, optimizer failures and the Poisson limit,
tied lift conservation/permutation invariance, chronological ages, and rendered
zero/constant/empty hexbins. The chapter flow test executes the actual labelled
cells on 30,010 synthetic policies with model/parameter fitters mocked to
inspect their inputs. Separate numerical tests exercise the real fitters.
All Phase 1 likelihood and tiny-tail regression tests still pass.

All 42 chapter Python cells compile. AST comparison confirms the six Phase 1
likelihood, dispersion, selection, and probability functions are unchanged
from the start of this session. Hash comparison verified all 26 other
previously modified files before the handoff refresh; only AGENTS.md was then
updated to point to remaining Phase 3 work. No figures or frozen outputs were
regenerated. The source, exports, chapter migration, and test diff were reviewed.
The unrestricted git diff --check still reports the pre-existing PDF whitespace
issue; the PDF is unchanged by this session.

## Phase 3 implemented and verified

Completed on 2026-09-06. The chapter now:

- Reads upper tails from survival probabilities: higher CDF means less
  probability above the same threshold. Gamma survival has exponential decay
  with a power factor; log-log curvature is expected.
- Separates outcome inequality (the normalization denominator) from
  prediction-ranked Gini. Raw Gini is twice the signed diagonal-minus-Lorenz
  area; area and midrank formulas agree on the same tied blocks.
- Uses tolerance-based permutation checks and the strictly increasing log
  transform of positive predictions, avoiding exponential overflow.
- Reads double lift against both model curves, explains low/high A/B ratios
  and the mean of individual ratios, and removes endpoint-anchoring claims.
- Explains the necessary claim-free residual stripe, additive versus
  multiplicative mean bias, and why weighted calibration/lift judge the mean.
  Neither symmetric residuals nor outcomes near the identity line are required.
- Deletes the positive-loss log-ratio panel, cell, and calibration claims.
  The retained positive-loss log-log outcome view is explicitly descriptive.
- Labels training diagnostics and the locked test audit, removes pooled-curve
  family-selection advice, and requires a new evaluation cycle for changes
  informed by the test audit. The unused chapter colour import is removed.

The existing suite includes the plan's Gini and double-lift counterexamples:
area 0.25/raw Gini 0.5/normalized Gini 1; constant and reversed scores;
fractional weights and tied permutations; increasing observations despite
exact B predictions; unanchored endpoints; and mean ratio 1.5 versus ratio
of means 5/3.

Full-output inspection exposed one additional label defect: print_frame
discarded unnamed nondefault indexes, hiding split and component names.
It now omits only an unnamed default row index and uses pandas reset_index
for meaningful labels. One regression check covers unnamed labels, named
labels, and default row numbering. No interfaces or dependencies were added.
A reconstructed pre-change SHA-256 confirms every other byte of the library,
including all modelling and diagnostic functions, is preserved.

### Phase 3 validation

~~~text
uv sync --locked --all-groups
Resolved 131 packages; audited 128 packages

uv run python -m pytest tests/ -q
43 passed, 132 subtests passed in 37.16s

uv run python -m pytest tests/ -q -k print_frame
1 passed, 42 deselected in 8.47s
(The final named-header assertion was strengthened after the full run.)

uv run ruff check .
All checks passed!

uv run ruff format --check .
7 files already formatted

git diff --check -- . ':!docs/compound-poisson-gamma-theory.pdf'
Passed
~~~

All 42 tutorial Python cells, 9 Poisson cells, and 10 Tweedie offset/weight
cells compile. All 61 computational cells also executed fresh successfully.
The index and theory chapters contain no Python cells.

The final CLI smoke used this exact command and completed with seven CSV
tables and four PNGs outside tracked artifacts:

~~~sh
uv run pure-premium --n-samples 10000 --n-alphas 2 --max-rounds 50 --early-stopping-rounds 5 --output-dir /tmp/nonlife-phase3-smoke > /tmp/nonlife-phase3-smoke.log 2>&1
~~~

### Fresh full-data execution

Quarto 1.9.38 used the project Python environment. Context7 and the
[Quarto execution documentation](https://quarto.org/docs/projects/code-execution.html#freeze)
confirmed that explicit single-document renders execute code even with
freeze: auto. Each chapter was rendered explicitly with caching disabled
and a restarted kernel. No freeze override was needed; book/_quarto.yml,
Cosmo, KaTeX, and the chapter structure are unchanged.

Final tutorial command after the row-label fix:

~~~sh
QUARTO_PYTHON=/home/bsatom/Documents/nonlife-pureprem/.venv/bin/python quarto render book/pure_premium_tutorial.qmd --no-cache --execute-daemon-restart --log /tmp/nonlife-phase3-tutorial-verified.log
~~~

The other chapters were each rendered fresh:

~~~sh
for chapter in index theory poisson_offset_weight tweedie_offset_weight; do
  QUARTO_PYTHON=/home/bsatom/Documents/nonlife-pureprem/.venv/bin/python quarto render "book/$chapter.qmd" --no-cache --execute-daemon-restart --log "/tmp/nonlife-phase3-$chapter.log" || exit
done
~~~

The final complete-book verification used the original freeze configuration:

~~~sh
QUARTO_PYTHON=/home/bsatom/Documents/nonlife-pureprem/.venv/bin/python quarto render book --no-cache --log /tmp/nonlife-phase3-book.log
~~~

That last render completed all five pages using the refreshed frozen results;
it did not execute cells again. All 15 local image references resolve.
The three computational frozen outputs contain no stderr outputs. Logs show
the Jupyter kernel's TCP transport warning at startup, with no execution or
convergence errors.

The final tutorial uses all 678,013 freMTPL2 policies, n_samples=None,
30 GLUM alpha candidates, up to 2,000 boosting rounds, and early stopping at
50 rounds. Seed 42 and the original splits remain fixed:

| Scope | Policies |
| --- | ---: |
| Development train | 433,928 |
| Development validation | 108,482 |
| Final train | 542,410 |
| Locked test | 135,603 |
| Eligible severity training policies | 19,955 |

Parameters use all eligible training rows before sampling. The CDF/SF samples
contain 20,000 frequency policies, all 19,955 eligible severity policies,
and 20,000 pure-premium policies. The separate uncapped training view contains
21,146 positive claim records. References use 128 thresholds and the existing
bounded batches. Selected Tweedie power is 1.5; fitted Gamma dispersion is
1.17155, Tweedie dispersion is 313.757, and NB shape is 1.17491 (rounded).

### Inspected results and artifacts

All 15 tutorial figures were visually inspected:

- frequency-cdf-output-2.png and frequency-sf-output-2.png: full zero-count
  atom, identical sample labels, matched observables, and labelled NB shape.
- severity-cdf-output-2.png and severity-sf-output-2.png: 19,955-policy labels,
  capped policy-average severity, Gamma dispersion, and tail curvature.
- pure-premium-cdf-output-2.png and pure-premium-sf-output-2.png: zero atom,
  capped totals, matching 20,000-policy samples, power/dispersion labels,
  and survival probabilities retaining the claim-free denominator.
- uncapped-training-claims-output-1.png: separately labelled descriptive
  per-claim tail with 21,146 training claims.
- conditional-dispersion-output-5.png: all eligible training rows, area labels,
  and fixed references 1, 1.17, and 314.
- cell-28-output-1.png: chronological driver-age intervals.
- cell-31-output-1.png, cell-34-output-1.png, and cell-39-output-1.png:
  single lift, Lorenz curves, and double lift with both model curves.
- cell-40-output-1.png, cell-41-output-1.png, and cell-42-output-1.png:
  linear outcomes including zero, descriptive positive-loss log-log outcomes,
  and the expected claim-free residual stripe with shared log-count colours.

All image hashes after the final row-label refresh match the already inspected
figures. The CLI smoke's constant-score GLUM Tweedie lift has flat observed
lift. Separate temporary plots verified zero outcomes, constant predictions,
empty visible bins, and an empty positive-only panel:
 /tmp/nonlife-phase3-hexbin-edge.png and
 /tmp/nonlife-phase3-hexbin-no-positives.png.

The Gamma/Tweedie working tails visibly depart from the empirical tails, and
area residual dispersions exceed their fitted references. These results do
not justify a family switch by themselves. In the locked test audit, LightGBM
frequency x severity has deviance 73.3357 and A/E 1.02016; direct LightGBM
Tweedie has deviance 73.6108 and A/E 1.11671. No recalibration was fitted to
those results. The raw/log prediction Gini check returns 0.327742 in both
cases. These are recorded results, not a deployment decision.

All three tracked computational html.json files are refreshed. The 15 current
PNG references exactly match the tutorial's frozen PNG files; obsolete source
reveals and the deleted panel are absent. Eleven unreferenced older PNGs
(cell numbers 7, 8, 9, 10, 13, 14, 15, 25, 36, 37, 38) were moved to
/tmp/nonlife-phase3-obsolete.EKky2J and remain recoverable there.
book/_book remains ignored and outside tracked changes.

An initial tutorial attempt was intentionally interrupted for source cleanup;
the earlier completed render exposed the printer defect. The acceptance log
is /tmp/nonlife-phase3-tutorial-verified.log, after that defect was fixed.
The final book log is /tmp/nonlife-phase3-book.log.

The pre-existing PDF still triggers the unrestricted git diff --check.
Its hash, and the hashes of unrelated prior README, other chapter sources,
theory source, exports, and Quarto configuration changes, were preserved.
The final phase diff was reviewed. AGENTS.md now points to completed phase
scope rather than remaining Phase 3 work.

## Remaining limits

The NB finite-shape search is bounded to [1e-8, 1e8], and improvements of at
most 1e-8 log-likelihood units per policy count as the Poisson limit. A material
boundary optimum raises. Gamma/Tweedie references remain fitted working
distributions for capped targets; pooled agreement cannot establish conditional
correctness. Synthetic checks do not establish full-data actuarial conclusions.

## Working-tree precautions

The repository already contained uncommitted changes to README, several book
chapters, theory source/PDF, and frozen outputs before Phase 1. Phase 1 changed
the main library module and chapter on top of that work, plus package exports
and tests. Hash comparison confirmed that the other 25 previously modified
files were preserved.

Re-read git status and the relevant diffs before further work. Do not reset,
clean, or overwrite unrelated changes. Do not rely on temporary snapshots in
/tmp surviving a context reset or a new environment.

The sandbox helper has intermittently failed with
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted.
Use the normal approval mechanism for a required command that fails this way;
do not mistake the helper failure for a source or dependency defect.

## Review handoff

All planned implementation and acceptance checks are complete. Review the
current uncommitted working tree before any further work. Publishing remains
outside this plan; no subsequent implementation phase was started.

## Phrasing audit edits (2026-09-06)

Applied the user-approved no-ai-slop findings to the source prose. Removed
repeated theory recaps, condensed the modeling reference into one table,
shortened the deviance walkthrough, reduced decorative emphasis, and replaced
prose em dashes. Retained the Python table placeholder and technical contrasts.

Validation: fenced code matches the pre-edit snapshot except for the approved
Poisson docstring punctuation. Quarto callout counts and display-math delimiter
balance checks passed. The diff whitespace check passed with the pre-existing
PDF excluded. Typst compiled successfully to a temporary PDF. The source diff
was reviewed against the no-ai-slop checklist.

The book, frozen results, and tracked PDF were not regenerated in this prose
pass. No statistical code changed; the computational tests were not rerun.

## Uncapped exploration before fitting (2026-09-06)

Implemented in labelled chapter cells, with regression coverage added to the
existing chapter-cell tests. No library code, public APIs, dependencies,
capped targets, or later conditional diagnostics changed in this phase.
The individual-claim tail moved before fitting and retains equal claim weights.

- All 542,410 training policies feed the exposure-weighted observed count
  proportions and matched Poisson mixture, including zero and >16 overflow.
- All 19,955 claim-bearing training policies (21,146 claims) feed uncapped
  policy-average severity. Its claim-weighted mean is EUR 2,346.72 and fitted
  base Gamma dispersion is 1.54079. Combining identical claim counts gives
  the exact claim-weighted mixture while avoiding a policies-by-grid array.
- Density uses logarithmic amount bins with density per euro. Direct Gamma
  survival uses full denominators and extends to the EUR 4,075,400.56 maximum.
  Zero probabilities are removed only for log display; the survival y-axis
  floor is two decades below the smallest positive empirical probability.
  Smaller positive Gamma values remain calculated and lie below the panel.
- Grouped V divides claim-weighted residual squares by claim-bearing policy
  count minus one. Tables report policy/claim counts and exclusions. Twelve
  vehicle-power groups qualify; BonusMalus excludes 23 sparse groups and one
  retained zero-variance group. Both intercept and slope are estimated with
  equal group weights. Descriptive slopes are 6.26327 and 4.97058. Empty or
  insufficient groups produce an explicit explanation instead of a fit.

The full-data interpretation records departures without a family-selection
claim. Exposure-weighted zero-count proportions are 0.951939 observed and
0.944352 Poisson. The constant-mean Gamma survival is above observed at
EUR 10,000 (0.0265315 versus 0.0173555), but below observed at EUR 100,000
(1.93296e-13 versus 0.00193890) and EUR 1,000,000 (6.92895e-122 versus
9.45805e-5). Pooled agreement would not establish conditional validity;
large losses and within-group risk differences affect the grouped slopes.

Validation completed:

~~~text
uv run python -m pytest tests/ -q
44 passed, 132 subtests passed in 94.97s
uv run ruff check .
All checks passed!
uv run ruff format --check .
7 files already formatted
~~~

The regression executes every new labelled cell before all fitting stages.
It verifies uncapped training inputs, absence of sampling, unequal exposure
and count mixtures, full denominators and tail endpoints, histogram weights,
the grouped formula, non-unit fitted intercept, sparse/zero-variance groups,
insufficient distinct means, and an entirely empty logarithmic panel. This
last case exposed and fixed a Matplotlib log-axis failure. Existing model,
likelihood, conditional diagnostic, and plotting checks still pass.

Final render, frozen-output and assembled-book verification: passed.
The preliminary render exited with signal 15 during LightGBM fitting without
a Python traceback; it is not the acceptance run. Full-data exploration was
also executed separately for figure inspection and numeric interpretation.
The acceptance render restarts the kernel and disables computation caching:

~~~sh
QUARTO_PYTHON=/home/bsatom/Documents/nonlife-pureprem/.venv/bin/python quarto render book/pure_premium_tutorial.qmd --no-cache --execute-daemon-restart --log /tmp/nonlife-uncapped-tutorial-verified.log
~~~

Single-document rendering executes code even with freeze: auto, as documented
in [Quarto's Python execution guide](https://quarto.org/docs/computations/python.html).
Model settings remain n_samples=None, n_alphas=30, max_rounds=2000,
early_stopping_rounds=50. The chapter ran all 48 cells with a restarted kernel.
A completed intermediate render was followed by a final full execution after
correcting the empirical survival steps: interval steps now retain the final
positive-survival interval below the maximum. The regression asserts the
actual step edges and values, not only the axis range.

The assembled book was verified with:

~~~sh
QUARTO_PYTHON=/home/bsatom/Documents/nonlife-pureprem/.venv/bin/python quarto render book --metadata-file /tmp/nonlife-uncapped-freeze.yml --log /tmp/nonlife-uncapped-book-verified.log
~~~

The temporary metadata contains only `execute: {freeze: true}`. It reuses
existing frozen results for the other chapters and the freshly verified main
chapter. No repository configuration was changed. Do not use --no-execute for
this assembly: in Quarto 1.9.38 it disables thawing as well as computation and
produces pages without displayed outputs. The HTML image-count check caught
that initial assembly attempt; the command above is the accepted result.

Visually inspected all four new figures and the moved individual-claim tail:

- exploratory-frequency-output-3.png: zero and overflow bins, all-policy
  labels, readable rotated integer ticks, and separate discrete points.
- exploratory-severity-density-output-1.png: logarithmic amount bins,
  claim-weighted density per euro, full observed range, and Gamma mixture.
- exploratory-severity-survival-output-1.png: linear-x/log-y and log-log
  panels with complete final empirical intervals, full maximum, and documented
  display floor. No denominator truncation or reference CDF subtraction.
- exploratory-severity-groups-output-5.png: separate grouping variables,
  Gamma reference and fitted intercept/slope curves, readable legends.
- uncapped-training-claims-output-1.png: distinct per-claim label, equal
  weighting over 21,146 claims, and the complete last interval.

The other 14 current tutorial figures match the pre-phase image hashes after
accounting for cell renumbering. Frozen references and PNG files agree exactly
at 19 figures. The new frozen cells precede power selection, contain the final
interval-step code, and retain all full-data outputs. All five assembled HTML
chapters exist, every local image resolves, and the tutorial contains all 19
figures. The artifact check is /tmp/nonlife-verify-uncapped.py.

Five unreferenced numbered PNGs (cells 28, 31, 39, 41, 42) were moved to
/tmp/nonlife-uncapped-obsolete and remain recoverable. The earlier interrupted
render's temporary notebook was also moved there. The book output directory
remains ignored. Hash comparison confirms unrelated source, library, PDF,
configuration, and other chapters' frozen files retain their pre-phase bytes.
The diff whitespace check passes excluding the pre-existing PDF; its existing
whitespace warning was not repaired in this phase.

PLAN.md records this phase's scope and acceptance checks. Review the current
uncommitted working tree; no next phase, commit, or publication was started.

## Post-fit count interpretation repair (2026-09-06)

Implemented the PLAN.md phase "Post-fit count interpretation repair". Two
prose edits in book/pure_premium_tutorial.qmd; no cells, library code, APIs,
figures, or frozen outputs changed.

- The paragraph after the conditional-dispersion cell no longer calls
  Pearson \(D_g=1.61\)–2.52 plus NB \(r=1.17\) joint overdispersion
  evidence. It now explains the disagreement: a single claim on a short
  exposure inflates the squared Pearson residual, NB2 at \(r=1.17\) raises
  the expected dispersion only by \(m/r\) (a few percent at typical policy
  means), the count survival panel places the two-or-more mass between the
  Poisson and NB references with both overstating the farther tail, and
  the diagnostics neither validate the Poisson PMF nor support a specific
  alternative. Poisson stays a working mean/deviance model. The Gamma and
  Tweedie sentences are unchanged; their \(D_g\gg\hat\phi\) reading already
  matches their too-fast survival tails.
- The 3.4 frequency pointer now reads "the later area residual-dispersion
  table" instead of "the Pearson residual check below".

A diff against book/pure_premium_tutorial.qmd.orig shows exactly three
prose hunks: the pre-session duplicated-word typo fix plus these two edits.
No fenced code block changed, so no re-render is required; the frozen
outputs and all 19 figures remain valid.

Validation completed:

~~~text
uv run python -m pytest tests/ -q
44 passed, 132 subtests passed in 44.86s

uv run ruff check .
All checks passed!

uv run ruff format --check .
7 files already formatted

git diff --check -- . ':!docs/compound-poisson-gamma-theory.pdf'
Passed
~~~

Stop here for review. Nothing committed or published.

## Chapter reading repair (2026-09-06)

Implemented the PLAN.md phase "Chapter reading repair". Kept the evaluation
stack, working-distribution split, and Poisson/NB interpretation from the
previous phase.

- Roadmap at the top of book/pure_premium_tutorial.qmd.
- `prepare_mtpl_data` now reports `claim_cap`; the prepare cell prints it.
- Exploration display notes cut to two lines; \(V_g\) equivalently clause
  removed.
- Tweedie Jacobian / GLUM density API moved into a collapsed callout with
  `show_source(select_tweedie_power)`.
- `RATING_FORMULA` shown in a collapsed callout.
- Gini main path is ranking, the integral, and the two-policy example.
  Midrank and software conventions sit in the existing callout. Lift
  callout is source only.
- Double-lift tables use the four plotted pairs; `combinations` import
  removed.
- "How to read" says driver-age quantile bands.
- Locked-test findings paragraph uses the recorded LightGBM numbers
  (deviance 73.34 / A/E 1.020 vs 73.61 / 1.117). No deployment claim.

No new figures. Cell changes: prepare audit keys, Tweedie `show_source`,
double-lift pair list. Tutorial freeze was already absent; this phase
does not re-render. All 46 chapter Python cells parse.

Validation completed:

~~~text
uv run python -m pytest tests/ -q
44 passed, 132 subtests passed in 38.73s

uv run ruff check .
All checks passed!

uv run ruff format --check .
7 files already formatted

git diff --check -- . ':!docs/compound-poisson-gamma-theory.pdf'
Passed
~~~

Stop here for review. Nothing committed or published.

## Conditional-distribution contrast (2026-09-07)

Implemented the PLAN.md phase "Conditional-distribution contrast". Three
prose edits in book/pure_premium_tutorial.qmd; no cells, library code, APIs,
figures, or frozen outputs changed.

- 3.4 now states that its pooled references fit no model and predict no
  policy, pointing to 3.7 for the prediction-based comparison.
- 3.7 opens with the contrast: 3.4 uses one constant portfolio mean, while
  3.7 compares observations with the GLUM predictions and the conditional
  distributions those predictions imply, under the same risk-volume weights.
- The "Fit parameters before sampling" subsection is retitled "Fit
  dispersion parameters; means stay fixed" and explains that only dispersion
  parameters are fitted; the Poisson frequency reference has nothing to fit.

The training_rows["frequency"]["ClaimNb"] selection in the diagnostic cell
was reviewed and deliberately left unchanged: it is a dict lookup followed by
a standard pandas column read; the chained-indexing hazard applies only to
assignment, and the row/prediction index alignment is correct.

A diff against the pre-edit snapshot shows exactly the three prose hunks
above; no fenced code block changed, so no re-render is required and the
frozen outputs and all 19 figures remain valid.

Validation completed:

~~~text
uv run python -m pytest tests/ -q
44 passed, 132 subtests passed in 33.94s

uv run ruff check .
All checks passed!

uv run ruff format --check .
8 files already formatted

git diff --check -- . ':!docs/compound-poisson-gamma-theory.pdf'
Passed
~~~

Stop here for review. Nothing committed or published.
