# Plan: Section 3.7 observed-vs-predicted clarification (prose-only)

Approved scope: prose-only. No Python cell, library, API, figure, or
frozen-output change, so no re-render (same precedent as the
"Post-fit count interpretation repair" phase). The
`training_rows["frequency"]["ClaimNb"]` selection stays as is: dict lookup
plus a standard column read, no chained-assignment hazard.

Edits are stated as exact old → new text for `book/pure_premium_tutorial.qmd`.
A pre-edit snapshot of the chapter is at
`/tmp/opencode/pure_premium_tutorial.qmd.pre` for the no-fenced-code-change
check.

## Edit 1: Section 3.4 forward pointer

After the paragraph ending "cannot establish conditional model validity or
select a model family." (before `### Frequency with unequal exposures`), add:

```
Nothing here fits a model or predicts a policy: each reference uses a single
constant portfolio mean, whereas section 3.7 repeats the comparison against
the fitted model's policy-specific predictions.
```

## Edit 2: Section 3.7 intro contrast paragraph

In `## Compare fitted conditional distributions on training policies`,
insert a first paragraph before "These post-fit training diagnostics use
capped targets and fitted policy means.":

```
Section 3.4 compared training outcomes with constant-mean portfolio
references: one rate for the whole portfolio, no conditional expectation, and
no per-policy prediction. This section compares observations with
predictions. The GLUM benchmark of the previous section supplies a
policy-specific mean to every eligible training row of each component, and
each reference below is the conditional distribution that prediction implies:
a Poisson or Negative Binomial count with mean $e_i\hat\lambda_i$, a Gamma
policy-average severity with mean $\hat\mu_i$, and a Tweedie capped policy
total with mean $e_i\hat\mu_i$. The panels therefore compare observations
with the predictions of the model fitted in section 3.6; the only quantities
fitted below are dispersion parameters.
```

The existing second paragraph is unchanged.

## Edit 3: Retitle and clarify the subsection

Replace:

```
### Fit parameters before sampling

Keep predicted rates fixed and fit the Gamma and Tweedie base dispersions and
Negative Binomial shape on every eligible training policy.
Only after those fits finish, sample up to 20,000 policies per component with
seed 42 and reuse each component's rows, predictions, and weights throughout.
```

with:

```
### Fit dispersion parameters; means stay fixed

No mean is estimated here. Predicted rates stay fixed; the only fitted
quantities are the Gamma base dispersion for severity, the Tweedie base
dispersion for pure premium, and the Negative Binomial shape for frequency,
each estimated on every eligible training row against the fixed predictions.
The Gamma and Tweedie references thus share the model's policy means and
working families with only a variance parameter estimated, and the Poisson
count reference needs nothing fitted at all: its policy means come straight
from the model.
Only after those fits finish, sample up to 20,000 policies per component with
seed 42 and reuse each component's rows, predictions, and weights throughout.
```

## Edit 4: PLAN.md

Update the "Updated:" date to 2026-09-07 and append:

```
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
```

## Edit 5: PROGRESS.md

Update the "Updated:" date to 2026-09-07 and append:

```
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

No fenced code block changed, so no re-render is required; the frozen
outputs and all 19 figures remain valid.

Validation completed:

~~~text
[fill in pytest, ruff, and diff-check results]
~~~

Stop here for review. Nothing committed or published.
```

## Validation

```sh
uv run python -m pytest tests/ -q
uv run ruff check .
uv run ruff format --check .
git diff --check -- . ':!docs/compound-poisson-gamma-theory.pdf'
diff /tmp/opencode/pure_premium_tutorial.qmd.pre book/pure_premium_tutorial.qmd
```

The diff must show only the three prose hunks (no fenced code block touched).
