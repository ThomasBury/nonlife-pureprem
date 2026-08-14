# French MTPL Frequency, Severity, and Pure Premium

This repository is an actuarial pricing tutorial built around the French
freMTPL2 data. The main tutorial follows the pricing workflow in order:

1. define positive paid claims and cap each claim at EUR 100,000;
2. model claim frequency;
3. model claim-weighted Gamma severity;
4. multiply frequency and severity into a decomposed pure premium;
5. model pure premium directly with Tweedie regression;
6. compare calibration, discrimination, lift, and portfolio balance on an
   untouched test set.

The library lives in `src/tweedie_regr/` (standard `src` layout, installed
editably by `uv sync`). The interactive tutorials are rendered as a
[Quarto book](https://quarto.org/) under `book/`, published at
[thomasbury.github.io/offset_weight_equivalence](https://thomasbury.github.io/offset_weight_equivalence/).

## Targets and Weights

| Component | Target | Sample weight | Rows |
| --- | --- | --- | --- |
| Frequency | ClaimNb / Exposure | Exposure | All policies |
| Severity | ClaimAmountCapped / ClaimNb | ClaimNb | Claim-bearing policies |
| Pure premium | ClaimAmountCapped / Exposure | Exposure | All policies |

The tutorial produces four final pure-premium predictions:

- GLUM frequency x GLUM severity;
- LightGBM frequency x LightGBM severity;
- direct GLUM Tweedie;
- direct LightGBM Tweedie.

GLUM provides formula-based spline/categorical models, regularized predictive
fits, full Tweedie likelihood for development selection of p, and separate
unregularized robust coefficient tables. LightGBM is the only nonlinear model
family and is tuned only on the internal development-validation split.

## Setup and Run

The project requires Python 3.14 and uses uv:

```powershell
uv sync
uv run pure-premium                # CLI entry point
uv run pure-premium --n-samples 10000  # quick smoke run
```

The OpenML datasets are downloaded on first run.

Run the tests and linters:

```powershell
uv run python -m pytest tests/ -v
uv run ruff check .
uv run ruff format --check .
```

## Quarto Book

The interactive tutorials live in `book/` as `.qmd` chapters. The Quarto CLI
is an external prerequisite and must be installed separately from the Python
environment. After `uv sync`, render from the repository root with Quarto
pinned to the project virtual environment.

POSIX:

```sh
QUARTO_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" quarto render book
```

PowerShell:

```powershell
$env:QUARTO_PYTHON = uv run python -c "import sys; print(sys.executable)"
quarto render book
```

`freeze: auto` remains enabled, so project renders rerun changed chapters.
Before merging a change that alters an executable chapter or its results, run
the full local render and commit the refreshed `book/_freeze/` results with it.
GitHub Pages renders with `--no-execute` and publishes those committed results.

Chapters cover the full pricing comparison, Poisson and Tweedie offset-weight
equivalence, and the underlying compound Poisson-Gamma theory.

For a fresh evaluation of the tutorial, do this from the repository root:

```sh
QUARTO_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" quarto render book --no-cache
```

## Outputs

By default, the main tutorial writes to artifacts/pure_premium:

- the Tweedie power profile;
- LightGBM tuning tables;
- compact GLUM coefficient, confidence-limit, and relativity tables;
- component_calibration.png;
- pure_premium_lorenz.png;
- pure_premium_lift.png;
- pure_premium_double_lift.png.

The coefficient intervals are classical robust inference conditional on the
chosen unregularized specification. They are not inference for the penalized
predictive fits.

Validation reports weighted deviance, D2, aggregate actual/expected, raw and
normalized Gini, and exposure-balanced lift deciles. The tutorial reports the
evidence without automatic model-superiority or universal calibration verdicts.

## Companion Material

The main pricing tutorial uses exposure as the rate-model weight under the
actuarial risk-volume convention. The Tweedie offset-weight chapter
([book/tweedie_offset_weight.qmd](book/tweedie_offset_weight.qmd))
derives the generic total-offset/rate-weight equivalence under a different
dispersion convention, where the exact Tweedie rate weight is
exposure^(2-p).

Focused companion chapters in the book:

- [book/poisson_offset_weight.qmd](book/poisson_offset_weight.qmd) — Poisson offset-weight equivalence
- [book/tweedie_offset_weight.qmd](book/tweedie_offset_weight.qmd) — Tweedie offset-weight equivalence
