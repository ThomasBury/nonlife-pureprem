# Offset–weight chapter cuts

Apply after review. No new figures, no new fits, no `boost_from_average=True` demo (verified: max abs diff vs `False` is 0 on this saturated model).

## `book/poisson_offset_weight.qmd`

Replace the intro after the numbered list with:

```
`predict()` returns a rate for (1) and (2). The custom objective returns a
raw log-rate, so (3) needs `exp`. Set `boost_from_average=False` on the
offset model: the offset already sets the level, and the default would add
`log(mean(count))` on top of `log(exposure)`.
```

Delete:

- `data_basic` frame, its `generate_claim_counts` call, and the whole `## Checks` section (N table + 40-row easy means)
- entire `## Solution 1B` section (`sol_1b_*` is fitted and never checked)
- the 40-row by-`lambda` loop in analysis (`for i in [1, 2, 3]: ... Solution {i} (Easy Data)`)
- trailing `data_easy.head()` cell

Keep: mismatch table, aggregate sums, 2×2 figure.

## `book/tweedie_offset_weight.qmd`

Replace the three-sentence common-dispersion disclaimer with:

```
::: {.callout-warning}
## This is not the ratemaking weight

This chapter uses a **common dispersion φ on totals**,
`S ~ Tweedie(e μ, φ)`. Rate weights are then `exposure**(2-p)`.

For pricing, the compound Poisson process has `Var(S) = e φ μ^p`, so the
pure-premium weight is `exposure`, for every `p`. See the theory chapter.
The two weights coincide only at `p = 1` (Poisson).
:::
```

Split the current "Equivalence Checks" table:

- Formulation: LGB offset vs LGB weighted only (RMSE + relative RMSE)
- Learner comparison: sklearn vs LGB offset, sklearn vs LGB weighted

Rewrite the figure paragraph so it does not hardcode `1.7e-9` / `9.2e-2`. Point at those two tables. Nested-marker sentence stays.

Delete:

- `import warnings`
- `dvalid` / `valid_sets` in both LightGBM fitters (`yte` in the offset fitter goes with them)
- entire `## Special cases: p=1 (Poisson) and p=2 (Gamma)` section (the callout already states the `p=1` coincidence)

## Verify

```
QUARTO_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" quarto render book/poisson_offset_weight.qmd
QUARTO_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" quarto render book/tweedie_offset_weight.qmd
```

Confirm: no `sol_1b` / `data_basic` / special-case tables; LGB pair still ~1e-9; figures still render.
