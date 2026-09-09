# Conditional diagnostics and chapter repair

Updated: 2026-09-07.

Read [PROGRESS.md](PROGRESS.md) for the handoff. Phases 1, 2, and 3 are
implemented and verified. The subsequent user-approved phase adds uncapped
exploration before fitting, as scoped below. Preserve earlier work; stop for
review after this phase and do not publish.

## Constraints

- Preserve existing uncommitted work. Review the working tree before editing;
  it includes earlier source, prose, PDF, and frozen-output changes.
- Backward compatibility is not required. Replace ambiguous interfaces,
  migrate every repository caller and package export, and delete the old
  interfaces and tests. Do not add compatibility aliases, wrappers, optional
  legacy inputs, deprecation paths, or fallback behavior to preserve them.
- Keep GLUM mean fitting and density evaluation. Reuse the Phase 1
  exposure-aware likelihood, dispersion estimator, and probability functions.
- Add no dependencies or speculative abstractions. Keep the existing Quarto
  theme and rendering setup.
- Update chapter calls and walkthroughs in the same phase as interface changes.
- Publishing is outside this plan. A phase ends at implementation, verification,
  and review.

## Main files

- [src/nonlife_pureprem/pure_premium.py](src/nonlife_pureprem/pure_premium.py):
  fitting, diagnostics, and shared evaluation helpers.
- [src/nonlife_pureprem/__init__.py](src/nonlife_pureprem/__init__.py):
  public exports.
- [tests/test_pure_premium_tutorial.py](tests/test_pure_premium_tutorial.py):
  extend the existing suite.
- [book/pure_premium_tutorial.qmd](book/pure_premium_tutorial.qmd):
  execution flow, plot calls, interpretation, and source walkthroughs.
- [book/_freeze/](book/_freeze/): refresh from fresh execution in Phase 3.

## Phase 2: Compare conditional distributions and repair plots

Status: implemented and verified on 2026-09-05; stopped for review.
See PROGRESS.md for final interfaces, validation, and remaining limits.

### 1. Replace pooled diagnostic inputs

Require predicted means aligned with the supplied rows. Gamma and Tweedie
plotters also require explicitly fitted base dispersion; the Negative Binomial
overlay requires its fitted shape. Plotters perform no fitting or sampling.
Trace every caller before changing signatures.

Use these observation-level references, where \(e_i\) is exposure,
\(n_i\) is claim count, and predictions are rates:

| Observable | Conditional distribution | Weight for empirical and reference curves |
| --- | --- | --- |
| Policy claim count | Poisson mean \(e_i\hat\lambda_i\) | Exposure \(e_i\) |
| Positive policy-average severity | Gamma shape \(n_i/\phi\), scale \(\phi\hat\mu_i/n_i\) | Claim count \(n_i\) |
| Capped policy total | Tweedie mean \(e_i\hat\mu_i\), dispersion \(\phi e_i^{1-p}\) | Exposure \(e_i\) |

Average each observation's CDF or survival probability with the same weights
as the empirical curve. Both curves in a panel must describe the same
observable and rows. Explain that these are fitted working distributions for
capped targets, and agreement with their pooled mixture does not establish
conditional correctness.

Add small, separate Gamma-dispersion and Negative-Binomial-shape fitters:

- Fit Gamma base dispersion with predicted severity means fixed, using the
  policy-average likelihood with claim-count scaling. Claim count already
  controls the observation's dispersion; do not multiply that likelihood by
  claim count again as a second fitting weight.
- Fit a positive real NB shape \(r\) against fixed predicted policy counts
  \(m_i=e_i\hat\lambda_i\), with success probability \(r/(r+m_i)\).
  Include an explicitly labelled Poisson limit when appropriate.
- Do not use scipy.stats.fit(nbinom, ...): its fitting interface constrains
  shape to integers. See the [SciPy fit documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fit.html).
- Check optimizer success and finite valid estimates; no silent moment
  fallback.

Once the diagnostic callers are migrated, delete _fit_tweedie_mle and its
legacy tests. Use fit_tweedie_dispersion for the Tweedie diagnostic flow.
Remove any imports made dead by these replacements.

### 2. Replace the area-power estimator

Replace mean_variance_table and mean_variance_diagnostic with
conditional_dispersion_table and conditional_dispersion_diagnostic.

By area, report row count, risk volume, observed and predicted rates, and

\[
D_g=\frac{1}{n_g}\sum_{i\in g}
w_i\frac{(y_i-\hat\mu_i)^2}{\hat\mu_i^p}.
\]

Here \(n_g\) is the number of eligible rows, risk volume is \(\sum w_i\),
and observed/predicted rates are weighted means using \(w_i\).
Fix \(p=1\) for frequency, \(p=2\) for severity, and the selected Tweedie
power for pure premium. Compare \(D_g\) with 1 for Poisson or the fitted base
dispersion for Gamma/Tweedie. Use eligible training rows.

Delete the area-based power fit, _INITIAL_VARIANCE_POWER, and curve_fit.
Remove their exports, walkthroughs, and family-selection advice. No aliases
for the removed functions.

### 3. Make the chapter diagnostic flow use training rows

1. Split immediately after preparation. Distinguish structural preparation
   audits from outcome summaries and exploration; use training rows for the
   latter.
2. Select Tweedie power on development data using the Phase 1 implementation.
3. Fit the training GLUM models before running conditional diagnostics.
4. Estimate all diagnostic dispersions and NB shape on all eligible training
   rows for that component.
5. Sample up to **20,000 eligible policies per component**, using seed **42**,
   only after parameter fitting. Keep predictions aligned with sampled rows.
6. Reuse each component's exact sample for its CDF and SF panels and for both
   empirical and reference curves. Label the sample size.
7. Evaluate **128 thresholds** in bounded batches; avoid a full
   thresholds-by-policies-by-series-terms allocation.
8. Retain uncapped per-claim tails as a separate descriptive view. Filter its
   claim records to training policy IDs. Do not present it as the fitted
   conditional distribution of capped policy-average severity.
9. Reserve the test split for the final locked audit.

### 4. Repair lift ties, age ordering, and hexbins

- Aggregate identical prediction scores or A/B ratios before cumulative
  interpolation in exposure_balanced_lift_table and
  exposure_balanced_double_lift_table. Distribute each tied block
  proportionally across equal-exposure bins.
- Preserve table columns, total exposure, and exposure-weighted observed and
  predicted totals. The prediction_ratio column remains the weighted
  average of individual A/B ratios.
- A constant prediction score must produce flat observed lift regardless of
  row order. Test ties that cross bin boundaries, not just unique scores.
- Keep driver-age bands as ordered categories through grouping and sorting.
  Convert to strings only when formatting tick labels.
- Linear hexbins must include zero on the observed axis and use nondegenerate
  limits for constant predictions.
- Handle empty visible bins and empty positive-only panels without failing
  in shared logarithmic colour normalization.

### Phase 2 acceptance gate

Extend the existing tests to cover:

- Heterogeneous Poisson exposures, Gamma claim counts, and Tweedie exposures
  against the specified conditional references and plotting weights.
- Alignment of rows and predictions; plotters do not fit or sample internally.
- Dispersion/shape fitting uses all eligible training rows, before sampling.
- Stable fixed-power residual dispersion across groups with different risk
  volumes; recover an NB shape below one and exercise the Poisson limit.
- Training-only policy IDs in diagnostics, including the uncapped claims view;
  deterministic sampling, the 20,000-row ceiling, and CDF/SF sample reuse.
- Row-permutation invariance and conservation across tied lift blocks.
- Chronological age bands and all-zero/constant-prediction/empty hexbins.
- All Phase 1 likelihood and tiny-tail regression tests still pass.

Run the full tests and Ruff checks below, review the complete phase diff,
record results and remaining limits in PROGRESS.md, then stop. The full-data
book execution belongs to Phase 3.

## Phase 3: Correct interpretation and regenerate the book

Status: implemented and verified on 2026-09-06; stopped for review.
See PROGRESS.md for full-data execution, inspected figures, and validation.

### 1. Correct the explanations and executable examples

- A higher CDF at a threshold means less probability above it. Use survival
  curves for upper-tail comparisons.
- Describe Gamma survival as exponential decay with a power factor
  asymptotically (a polynomial factor for integer shape); curvature on
  log-log axes is expected.
- Separate outcome inequality from prediction-ranked Gini. The Gini obtained
  by ordering on observed outcomes is the normalization denominator.
  Prediction-ranked Gini is **twice** the signed area between the diagonal
  and the tie-aware Lorenz curve:
  \(G=2\int_0^1[u-L(u)]\,du\).
- Replace exact permutation assertions with tolerance-based comparisons.
  Use np.log(raw) for the strictly positive prediction invariance example,
  avoiding overflow from np.exp(raw).
- Read double lift by comparing observed levels with **both** model curves.
  Low A/B ratios mean A predicts less relative to B; high ratios mean A
  predicts more relative to B. Rising observations alone do not favor A.
- Explain that prediction_ratio is a weighted average of individual ratios,
  generally different from the ratio of the two group means. Remove claims
  that lift endpoints are anchored.
- Claim-free policies necessarily form the residual stripe at
  \(-\hat\mu\). Use weighted calibration and lift tables to judge the mean;
  do not require symmetric residuals or individual outcomes near the identity
  line.
- For multiplicative bias \(E[Y\mid X]=c\hat\mu\), the expected residual is
  \((c-1)\hat\mu\). A constant expected residual offset is additive.
  Recalibration informed by evaluation requires a new evaluation cycle.
- Delete the positive-loss log-ratio panel, its executable cell, and its
  severity-calibration claims and checklist references. Retain the
  positive-loss log-log outcome view as descriptive.
- Remove remaining family-selection advice based on pooled curves or
  area-power estimates. Clearly label training diagnostics and the final
  test audit.
- Reconcile all function walkthroughs, source reveals, labels, imports,
  cross-references, and final checklists with the implemented interfaces.

The original plan referenced supplied Gini and double-lift counterexamples
but did not include their numeric inputs. These self-contained examples
preserve the intended checks:

- Equal weights, outcomes [0, 2], and ascending scores [1, 2] give signed
  area 0.25, raw Gini 0.5, and normalized Gini 1. Constant scores give raw
  Gini 0; permuting tied rows must not alter the result within tolerance.
- Equal weights, observed [1, 2, 4], A predictions [0.5, 2, 8], and B
  predictions [1, 2, 4] have increasing A/B ratios and observed rates.
  B nevertheless matches observed group levels exactly. A's endpoint
  predictions are not forced to equal observations.
- In one equal-weight bin, A [1, 4] and B [1, 2] give mean individual
  ratio 1.5, while the ratio of group means is \(5/3\).

Add these checks to the existing suite where they exercise executable
behavior, and use them to audit the corresponding prose.

### 2. Run the CLI and rebuild from fresh full-data computation

- Run a small CLI smoke test with outputs outside tracked artifacts:

  ~~~sh
  uv run pure-premium --n-samples 10000 --n-alphas 2 --max-rounds 50 --early-stopping-rounds 5 --output-dir /tmp/nonlife-phase3-smoke
  ~~~

- Execute the book using the full dataset and the intended model settings;
  a smoke sample is not evidence for the chapter's actuarial conclusions.
  Keep the diagnostic sample limits from Phase 2.
- Use the project Python environment. Check current Quarto execution options
  when carrying out this step. The current configuration is freeze: auto;
  explicitly ensure that every chapter runs fresh. Do not assume that a
  cache flag alone proves frozen results were bypassed.
- If a temporary freeze/cache override is needed, restore the existing
  configuration afterward. Keep the Cosmo theme, KaTeX setup, and chapter
  structure.
- Inspect execution logs and changed figures: zero mass, conditional CDF/SF
  alignment, tail scales, sample-size labels, dispersion references, tied
  lift behavior, chronological ages, and zero/constant/empty hexbin handling.
- Refresh tracked book/_freeze/ outputs from that successful run. Remove
  obsolete outputs belonging to deleted cells where appropriate, preserving
  unrelated existing work. Keep book/_book/ out of tracked changes.
- Verify the final render and record the exact execution commands, data
  scope, checks, and inspected figures in PROGRESS.md. Do not publish.

### Phase 3 acceptance gate

Run the full tests, Ruff lint/format checks, CLI smoke, fresh full-data book
execution, figure inspection, and diff review. Confirm no stale compatibility
interfaces, obsolete walkthroughs, positive-loss log-ratio claims, or stale
frozen outputs remain. Update PROGRESS.md and stop for review.

## Shared validation commands

Run from the repository root:

~~~sh
uv sync --locked --all-groups
uv run python -m pytest tests/ -q
uv run ruff check .
uv run ruff format --check .
git diff --check
~~~

The Phase 1 handoff records a pre-existing modified PDF that Git treats as
text and flags for whitespace. Do not modify that PDF to satisfy a text
whitespace check. Report it separately and check all other changes with:

~~~sh
git diff --check -- . ':!docs/compound-poisson-gamma-theory.pdf'
~~~

These checks complement the statistical and plotting acceptance tests; a
successful render or smoke run alone does not establish model validity.

## Uncapped exploration before fitting (2026-09-06)

Status: implemented and verified; stopped for review. See PROGRESS.md for
full-data findings, figure inspection, and exact validation commands.

One reviewable phase in the chapter and its existing cell tests. Add no
public APIs or dependencies; preserve capped modelling targets, later
conditional diagnostics, and unrelated working-tree changes.

- After training summaries and before power selection or mean fitting, use
  all training policies for exposure-weighted observed counts and a matched
  Poisson(exposure * total claims / total exposure) mixture. Show linear and
  log-y panels, zero claims, and an explicit overflow bin.
- For claim-bearing policies use ClaimAmountOriginal / ClaimNb, its
  claim-weighted constant mean, and fit_gamma_dispersion. Mix Gamma(count/phi,
  phi * mean/count) densities and direct survival probabilities with claim
  weights. Show logarithmic-bin density, linear-x/log-y and log-log survival
  through the observed maximum with full denominators. Keep the separate
  equal-weight individual-claim tail in this section.
- For VehPower and BonusMalus calculate the claim-weighted mean and
  V = sum(count * (severity - group mean)**2) / (claim-bearing policies - 1).
  Report policy and claim counts and excluded groups. Require at least four
  claims and two policies; omit zero variance from logarithmic fitting.
  Overlay phi * mean**2 and an unweighted log-log regression estimating both
  intercept and slope. Omit that regression with an explanation unless at
  least three valid groups and distinct means remain.
- Explain the common-mean Gamma interpretation and the effects of sparse
  groups, large losses, and risk heterogeneity. These exploratory plots and
  descriptive slopes do not establish conditional validity or choose the
  later family or Tweedie power. Report full-data findings after inspection.

Acceptance: extend chapter-cell checks for training-only uncapped inputs,
unequal exposures/counts, mixture weights, grouped formula, sparse and
zero-variance groups, omitted regressions, tail coverage and execution order.
Run the full tests and Ruff checks. Execute the full-data chapter afresh,
inspect figures, refresh its tracked frozen outputs, and assemble the book.
Record validation and stop for review without publishing.

## Post-fit count interpretation repair (2026-09-06)

Status: implemented; stopped for review.

Review of the uncapped-exploration phase found one remaining defect: the
post-fit count interpretation treated the area Pearson dispersion
1.61–2.52 and the fitted NB shape \(r=1.17\) as joint overdispersion
evidence. They disagree in magnitude and direction: NB2 at \(r=1.17\)
raises the expected Pearson dispersion only by \(m/r\), a few percent at
typical policy means, and the count survival panel shows both Poisson and
NB overstate the farther tail.

- Chapter-only prose repair in book/pure_premium_tutorial.qmd. Replace the
  "credible evidence of residual overdispersion" paragraph with the
  consistent reading: short exposures inflate squared Pearson residuals,
  \(r=1.17\) is a mild overlay, Poisson remains a working mean/deviance
  model, and no specific alternative count family is supported.
- Reword the 3.4 frequency pointer to name the later area
  residual-dispersion table instead of a generic "check below".
- No Python cells, library code, public APIs, frozen outputs, or figures
  change. The earlier prune (power regression, CDF panels, optimizer
  details) was already implemented and tested; do not repeat it.

Acceptance: full tests, Ruff checks, and the diff whitespace check
excluding the pre-existing PDF; confirm no cell code changed. No re-render:
prose between unchanged cells does not alter outputs.

## Chapter reading repair (2026-09-06)

Status: implemented; stopped for review.

Cuts and a short locked-test close. No new plots, models, or diagnostics.
Keep the split protocol, targets, four-model comparison, working-distribution
honesty, residual stripe, double-lift counterexample, and collapsed source.

- Add a four-line roadmap at the top.
- Print `claim_cap` in the preparation audit (`prepare_mtpl_data` report).
- Trim exploration display notes; drop the opaque \(V_g\) "equivalently"
  clause.
- Keep Tweedie power selection as development-likelihood only; move
  Jacobian / GLUM density details into a collapsed callout.
- Show `RATING_FORMULA` in a collapsed callout.
- Slim the Gini lecture; keep the two-policy example and permutation
  assert; move midrank and software conventions into the existing callout.
- Drop the numbered lift walkthrough; keep `show_source`.
- Print double-lift tables only for the four plotted pairs.
- Say "driver-age quantile bands", not "chronological bands".
- After the test charts, record this run: LightGBM frequency × severity
  deviance 73.34 and A/E 1.020; LightGBM Tweedie 73.61 and 1.117; no
  deployment claim.

Acceptance: full tests, Ruff, diff whitespace check excluding the
pre-existing PDF. No full-data re-render in this phase; freeze remains
the prior missing/out-of-scope state.

## Conditional-distribution contrast (2026-09-07)

Status: implemented; stopped for review.

Junior-reader clarification, chapter prose only. Sections 3.4 and 3.7 both
plot observed outcomes against Poisson references; the chapter did not state
the contrast.

- Section 3.4 gains one sentence: its pooled references fit no model and
  predict no policy, pointing to 3.7 for the prediction-based comparison.
- Section 3.7 opens with that contrast: 3.4 uses one constant portfolio
  mean, while 3.7 compares observations with the GLUM predictions and the
  conditional distributions those predictions imply, under the same
  risk-volume weights.
- The subsection "Fit parameters before sampling" is retitled "Fit
  dispersion parameters; means stay fixed" and states that only Gamma phi,
  Tweedie phi, and NB shape are fitted; the Poisson frequency reference has
  nothing to fit.

No Python cell, library, API, figure, or frozen-output change. Prose between
unchanged cells does not alter outputs, so no re-render. Acceptance: full
tests, Ruff checks, diff whitespace check excluding the pre-existing PDF.
