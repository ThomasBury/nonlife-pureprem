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

The main implementation is
[scripts/pure_premium/pure_premium.py](scripts/pure_premium/pure_premium.py).
It is import-safe: downloads and fitting happen only through main().

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
uv run python scripts/pure_premium/pure_premium.py
```

The OpenML datasets are downloaded on first run. For a reduced local smoke run:

```powershell
uv run python scripts/pure_premium/pure_premium.py --n-samples 10000 --n-alphas 3 --max-rounds 20 --early-stopping-rounds 5 --output-dir artifacts/smoke
```

Run the network-free checks with:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run python -m unittest discover -s tests -v
```
## Interactive Window

Open [scripts/pure_premium/pure_premium_interactive.py](scripts/pure_premium/pure_premium_interactive.py)
in VS Code, select the project kernel, then use **Run Current File in
Interactive Window**. The `# %%` cells use the complete data by default and
render the Rich tables plus temporary figures inline.

The companion distinguishes accuracy within response family, ranking via Gini
and low-to-high Lorenz dominance, portfolio calibration, and exposure-balanced
single and head-to-head double lift.

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
actuarial risk-volume convention. The companion note
[docs/compound_poisson_tweedie_pure_premium.md](docs/compound_poisson_tweedie_pure_premium.md)
derives the generic total-offset/rate-weight equivalence under a different
dispersion convention, where the exact Tweedie rate weight is
exposure^(2-p). The distinction is linked here instead of re-derived in the
main tutorial.

Focused companion scripts remain available:

- [scripts/offset_weight_eq_generic_poisson_tweedie/poisson_sim.py](scripts/offset_weight_eq_generic_poisson_tweedie/poisson_sim.py)
- [scripts/offset_weight_eq_generic_poisson_tweedie/tweedie_sim.py](scripts/offset_weight_eq_generic_poisson_tweedie/tweedie_sim.py)
- [scripts/pure_premium/tweedie_poisson_offset_from_average.py](scripts/pure_premium/tweedie_poisson_offset_from_average.py)

The companion documents remain mathematically independent of the main
frequency/severity pricing workflow.
