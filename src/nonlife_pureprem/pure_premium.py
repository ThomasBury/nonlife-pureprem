"""French MTPL pricing tutorial: frequency, severity, and pure premium.

The tutorial fits four final pure-premium estimates:

1. GLUM Poisson frequency x claim-weighted GLUM Gamma severity.
2. LightGBM Poisson frequency x claim-weighted LightGBM Gamma severity.
3. Direct GLUM Tweedie pure premium.
4. Direct LightGBM Tweedie pure premium.

All rate models use the actuarial risk-volume convention:

================  ===============================  ============
Model             Target                           Weight
================  ===============================  ============
Frequency         ClaimNb / Exposure               Exposure
Severity          ClaimAmountCapped / ClaimNb      ClaimNb
Pure premium      ClaimAmountCapped / Exposure     Exposure
================  ===============================  ============

The generic offset/weight identity under another dispersion convention is kept
in book/theory.qmd.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any, NamedTuple

import lightgbm as lgb
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from glum import (
    GeneralizedLinearRegressor,
    GeneralizedLinearRegressorCV,
    TweedieDistribution,
)
from rich.console import Console
from rich.table import Table
from scipy.optimize import curve_fit, minimize, minimize_scalar
from scipy.special import gammaln, logsumexp
from scipy.stats import fit, gamma, genpareto, nbinom, poisson
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_tweedie_deviance,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
CLAIM_CAP = 100_000.0
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
LEARNING_RATE = 0.05
REG_LAMBDA = 1.0
NUM_BOOST_ROUND = 2_000
EARLY_STOPPING_ROUNDS = 50
TWEEDIE_POWERS = tuple(float(p) for p in np.arange(1.1, 2.0, 0.1).round(1))
CONSOLE = Console()

# Okabe-Ito palette shared by every diagnostic plot. These mirror the
# tutorial's MODEL_COLORS ("observed" → black, "GLUM" → orange) so the
# library charts stay visually consistent without needing to pass colors
# through every helper.
_EMPIRICAL_COLOR = "#000000"
_FIT_COLOR = "#E69F00"
_FIT_COLOR_2 = "#56B4E9"
_FIT_COLOR_3 = "#009E73"

CATEGORICAL_FEATURES = ["VehBrand", "VehPower", "VehGas", "Region", "Area"]
LIGHTGBM_FEATURES = [
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "LogDensity",
    *CATEGORICAL_FEATURES,
]
RATING_FORMULA = (
    "C(VehBrand) + C(VehPower) + C(VehGas) + C(Region) + C(Area)"
    " + bs(VehAge, df=5, lower_bound=0, upper_bound=100)"
    " + bs(DrivAge, df=5, lower_bound=18, upper_bound=100)"
    " + bs(BonusMalus, knots=[55, 65, 80, 100],"
    " lower_bound=0, upper_bound=230) + LogDensity"
)


class TargetSpec(NamedTuple):
    target: str
    weight: str
    family: str
    claims_only: bool


TARGET_SPECS = {
    "frequency": TargetSpec("Frequency", "Exposure", "poisson", False),
    "severity": TargetSpec("Severity", "ClaimNb", "gamma", True),
    "pure_premium": TargetSpec("PurePremium", "Exposure", "tweedie", False),
}


class DataSplits(NamedTuple):
    train: pd.DataFrame
    test: pd.DataFrame
    development_train: pd.DataFrame
    development_validation: pd.DataFrame


def target_weight_table() -> pd.DataFrame:
    """Return the three coherent target/weight definitions used in the tutorial."""
    return pd.DataFrame(
        [
            {
                "model": name,
                "target": spec.target,
                "weight": spec.weight,
                "family": spec.family,
                "rows": "claims only" if spec.claims_only else "all policies",
            }
            for name, spec in TARGET_SPECS.items()
        ]
    )


def prepare_mtpl_data(
    frequency: pd.DataFrame,
    severity: pd.DataFrame,
    claim_cap: float = CLAIM_CAP,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float]]:
    """Validate source data, cap individual claims, aggregate, and derive targets.

    Returns ``(policies, claims, report)``:

    - ``policies`` is one row per policy with the audit-ready targets
      (``Frequency``, ``Severity``, ``PurePremium``).
    - ``claims`` is one row per individual claim with the raw and capped
      amounts; it is what uncapped empirical CCDFs use to show the true tail.
    - ``report`` is the audit dict used by the tutorial's preparation table.
    """
    required_frequency = {"IDpol", "ClaimNb", "Exposure", "Density"}
    required_severity = {"IDpol", "ClaimAmount"}
    missing_frequency = required_frequency.difference(frequency.columns)
    missing_severity = required_severity.difference(severity.columns)
    if missing_frequency or missing_severity:
        raise ValueError(
            f"Missing columns: frequency={sorted(missing_frequency)}, "
            f"severity={sorted(missing_severity)}"
        )
    if not np.isfinite(claim_cap) or claim_cap <= 0:
        raise ValueError("claim_cap must be finite and positive")

    policies = frequency.copy()
    policies["IDpol"] = policies["IDpol"].astype(int)
    if policies["IDpol"].duplicated().any():
        raise ValueError("Frequency data must contain one row per IDpol")
    policies = policies.set_index("IDpol")

    for column in ("Exposure", "Density"):
        values = pd.to_numeric(policies[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError(f"{column} must contain only finite positive values")
        policies[column] = values

    exposure_capped_rows = int((policies["Exposure"] > 1).sum())
    policies["Exposure"] = policies["Exposure"].clip(upper=1)
    policies = policies.rename(columns={"ClaimNb": "SourceClaimNb"})
    policies["SourceClaimNb"] = pd.to_numeric(
        policies["SourceClaimNb"], errors="raise"
    ).astype(int)

    claims = severity.copy()
    claims["IDpol"] = claims["IDpol"].astype(int)
    claims = claims[claims["IDpol"].isin(policies.index)]
    amounts = pd.to_numeric(claims["ClaimAmount"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.all(np.isfinite(amounts)):
        raise ValueError("ClaimAmount must contain only finite values")
    claims = claims.loc[amounts > 0].copy()
    claims["ClaimAmount"] = amounts[amounts > 0]
    claims["ClaimAmountCapped"] = claims["ClaimAmount"].clip(upper=claim_cap)
    per_claim = claims[["IDpol", "ClaimAmount", "ClaimAmountCapped"]].reset_index(
        drop=True
    )

    aggregated = claims.groupby("IDpol").agg(
        ClaimNb=("ClaimAmount", "size"),
        ClaimAmountOriginal=("ClaimAmount", "sum"),
        ClaimAmountCapped=("ClaimAmountCapped", "sum"),
    )
    policies = policies.join(aggregated, how="left")
    amount_columns = ["ClaimNb", "ClaimAmountOriginal", "ClaimAmountCapped"]
    policies[amount_columns] = policies[amount_columns].fillna(0)
    policies["ClaimNb"] = policies["ClaimNb"].astype(int)

    policies["Frequency"] = policies["ClaimNb"] / policies["Exposure"]
    policies["Severity"] = np.where(
        policies["ClaimNb"] > 0,
        policies["ClaimAmountCapped"] / policies["ClaimNb"],
        np.nan,
    )
    policies["PurePremium"] = policies["ClaimAmountCapped"] / policies["Exposure"]
    policies["LogDensity"] = np.log(policies["Density"])

    for column in CATEGORICAL_FEATURES:
        if column in policies:
            policies[column] = (
                policies[column].astype("string").str.strip("'").astype("category")
            )

    original_total = float(policies["ClaimAmountOriginal"].sum())
    capped_total = float(policies["ClaimAmountCapped"].sum())
    report: dict[str, int | float] = {
        "policies": len(policies),
        "positive_claims": int(policies["ClaimNb"].sum()),
        "exposure_capped_rows": exposure_capped_rows,
        "claim_count_mismatch_rows": int(
            (policies["SourceClaimNb"] != policies["ClaimNb"]).sum()
        ),
        "original_claim_amount": original_total,
        "capped_claim_amount": capped_total,
        "capped_loss_share": (
            (original_total - capped_total) / original_total
            if original_total > 0
            else 0.0
        ),
    }
    return policies, per_claim, report


def load_mtpl2(
    n_samples: int | None = None,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float]]:
    """Fetch freMTPL2 frequency/severity data and prepare a reproducible sample.

    Returns ``(data, claims, audit)``; see :func:`prepare_mtpl_data` for what
    ``data`` (per-policy) and ``claims`` (per-claim, uncapped) hold.
    """
    frequency = fetch_openml(data_id=41214, as_frame=True).data
    severity = fetch_openml(data_id=41215, as_frame=True).data
    if n_samples is not None:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        frequency = frequency.sample(
            n=min(n_samples, len(frequency)), random_state=random_state
        )
    return prepare_mtpl_data(frequency, severity)


def make_splits(
    data: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> DataSplits:
    """Create an untouched test set and a separate tuning-validation set."""
    claim_presence = data["ClaimNb"].gt(0)
    train, test = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=claim_presence,
    )
    development_train, development_validation = train_test_split(
        train,
        test_size=VALIDATION_SIZE,
        random_state=random_state,
        stratify=train["ClaimNb"].gt(0),
    )
    return DataSplits(train, test, development_train, development_validation)


def target_and_weight(
    data: pd.DataFrame,
    component: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return the rows, target, and weights for one pricing component."""
    spec = TARGET_SPECS[component]
    rows = data.loc[data["ClaimNb"] > 0] if spec.claims_only else data
    return (
        rows,
        rows[spec.target].to_numpy(dtype=float),
        rows[spec.weight].to_numpy(dtype=float),
    )


def weighted_empirical_cdf(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted ``values`` and their weighted empirical CDF."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if np.any(weights <= 0) or weights.sum() <= 0:
        raise ValueError("weights must be positive")
    order = np.argsort(values, kind="stable")
    return values[order], np.cumsum(weights[order]) / weights.sum()


def weighted_empirical_ccdf(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted ``values`` and their weighted empirical CCDF (1 − F̂)."""
    sorted_x, cdf = weighted_empirical_cdf(values, weights)
    return sorted_x, 1.0 - cdf


def _cdf_diagnostic_axes(
    axes: tuple[plt.Axes, plt.Axes] | None, figsize: tuple[float, float]
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig = axes[0].figure
    return fig, axes


def _single_panel_axes(
    axes: plt.Axes | None, figsize: tuple[float, float]
) -> tuple[plt.Figure, plt.Axes]:
    if axes is None:
        fig, axes = plt.subplots(figsize=figsize)
    else:
        fig = axes.figure
    return fig, axes


def _set_integer_xaxis(axis: plt.Axes, x_max: int) -> None:
    """Place ticks at integer positions on a discrete-support axis."""
    axis.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
    axis.set_xlim(left=-0.5, right=x_max + 1.5)


def _set_log_y_observed_floor(
    axis: plt.Axes, empirical_min: float, factor: float = 100.0
) -> None:
    """Pin the log-y lower bound to the smallest observed probability.

    The fitted line can drop far below the empirical when the family is too
    light-tailed for the data; pinning the bound to the empirical minimum
    keeps the tail visible. ``factor`` defaults to two decades of headroom
    below the smallest observed probability.
    """
    if empirical_min > 0:
        axis.set_ylim(bottom=float(empirical_min) / factor)


def _fit_tweedie_mle(
    y: np.ndarray,
    tweedie_power: float,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, float]:
    """Estimate (μ̂, φ̂) by profile maximum likelihood at a fixed power.

    Profile MLE: scan over μ on the log scale; at each μ solve φ from Pearson
    residuals via ``TweedieDistribution.dispersion``; evaluate glum's series
    Tweedie log-likelihood at (μ, φ(μ)); pick the μ that maximizes it. φ̂ is
    re-evaluated at μ̂ for the return value. This is the standard GLM recipe
    and matches what ``select_tweedie_power`` does internally, so the
    diagnostic uses the same fit as the rest of the tutorial.

    Falls back to the moment-based estimator (μ̂ = weighted mean,
    φ̂ = Var / μ̂^p) when the optimizer cannot return a finite value (all-zero
    data, non-finite μ_init, etc.) so an exotic dataset cannot break the
    diagnostic.
    """
    family = TweedieDistribution(tweedie_power)
    y_arr = np.asarray(y, dtype=float)
    w_arr = (
        np.asarray(sample_weight, dtype=float)
        if sample_weight is not None
        else np.ones_like(y_arr)
    )
    mu_init = float((y_arr * w_arr).sum() / w_arr.sum())

    def moment_fallback() -> tuple[float, float]:
        variance = float(((y_arr - mu_init) ** 2 * w_arr).sum() / w_arr.sum())
        phi = variance / mu_init**tweedie_power if mu_init > 0 else float("nan")
        return mu_init, phi

    if mu_init <= 0:
        return moment_fallback()

    def neg_log_likelihood(log_mu: float) -> float:
        mu = float(np.exp(log_mu))
        pred = np.full_like(y_arr, mu)
        phi = family.dispersion(y=y_arr, mu=pred, sample_weight=w_arr, ddof=0)
        return -family.log_likelihood(
            y=y_arr, mu=pred, sample_weight=w_arr, dispersion=phi
        )

    try:
        result = minimize_scalar(
            neg_log_likelihood,
            bracket=(
                np.log(mu_init * 0.5),
                np.log(mu_init),
                np.log(mu_init * 2.0),
            ),
        )
    except ValueError:
        return moment_fallback()
    mu_hat = float(np.exp(result.x))
    if not np.isfinite(mu_hat):
        return moment_fallback()
    phi_hat = float(
        family.dispersion(
            y=y_arr,
            mu=np.full_like(y_arr, mu_hat),
            sample_weight=w_arr,
            ddof=0,
        )
    )
    if not np.isfinite(phi_hat):
        return moment_fallback()
    return mu_hat, phi_hat


def _set_upper_x_bound(axis: plt.Axes, x_max: float) -> None:
    """Pin the upper x-axis bound to a quantile-based cap.

    The cap drops the small-mass extreme that would otherwise stretch a log-x
    axis over multiple decades and squash the bulk.
    """
    axis.set_xlim(right=float(x_max))


def poisson_cdf_diagnostic(
    data: pd.DataFrame,
    axes: tuple[plt.Axes, plt.Axes] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Compare the weighted empirical ClaimNb CDF to Poisson(λ̂) on linear and log-y axes."""
    fig, axes = _cdf_diagnostic_axes(axes, (11.0, 4.5))
    lambda_hat = float(data["ClaimNb"].sum() / data["Exposure"].sum())
    empirical_x, empirical_f = weighted_empirical_cdf(
        data["ClaimNb"].to_numpy(dtype=float),
        data["Exposure"].to_numpy(dtype=float),
    )
    x_grid = np.arange(int(empirical_x.max()) + 2)
    theoretical_f = poisson.cdf(x_grid, lambda_hat)
    x_max = int(empirical_x.max())
    for axis, log_y in zip(axes, (False, True)):
        axis.step(
            empirical_x,
            empirical_f,
            where="post",
            color=_EMPIRICAL_COLOR,
            label="Empirical",
        )
        axis.step(
            x_grid,
            theoretical_f,
            where="post",
            color=_FIT_COLOR,
            label=f"Poisson(λ={lambda_hat:.4f})",
        )
        axis.set_xlabel("ClaimNb")
        axis.set_ylabel("CDF")
        axis.set_title("Claim count" + (" (log-y)" if log_y else " (linear)"))
        if log_y:
            axis.set_yscale("log")
            _set_log_y_observed_floor(axis, empirical_f.min())
        _set_integer_xaxis(axis, x_max)
        axis.legend()
    fig.tight_layout()
    return fig, axes


def poisson_ccdf_diagnostic(
    data: pd.DataFrame,
    axes: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Compare the weighted empirical ClaimNb CCDF to Poisson, Negative Binomial

    NB absorbs overdispersion (``Var[N] = μ + α·μ²``); 
    """
    fig, axis = _single_panel_axes(axes, (6.5, 4.5))
    lambda_hat = float(data["ClaimNb"].sum() / data["Exposure"].sum())
    counts = data["ClaimNb"].to_numpy(dtype=int)
    nbinom_fit = fit(nbinom, counts, bounds={"n": (0.0, 100.0)})
    n_nbinom, p_nbinom = float(nbinom_fit.params.n), float(nbinom_fit.params.p)
    empirical_x, empirical_s = weighted_empirical_ccdf(
        data["ClaimNb"].to_numpy(dtype=float),
        data["Exposure"].to_numpy(dtype=float),
    )
    x_grid = np.arange(int(empirical_x.max()) + 2)
    poisson_s = poisson.sf(x_grid, lambda_hat)
    nbinom_s = nbinom.sf(x_grid, n_nbinom, p_nbinom)
    axis.step(
        empirical_x,
        empirical_s,
        where="post",
        color=_EMPIRICAL_COLOR,
        label="Empirical",
    )
    axis.step(
        x_grid,
        poisson_s,
        where="post",
        color=_FIT_COLOR,
        label=f"Poisson(λ={lambda_hat:.4f})",
    )
    axis.step(
        x_grid,
        nbinom_s,
        where="post",
        color=_FIT_COLOR_2,
        label=f"NB(n={n_nbinom:.2f}, p={p_nbinom:.3f})",
    )
    axis.set_yscale("log")
    _set_log_y_observed_floor(axis, empirical_s.min())
    axis.set_xlabel("ClaimNb")
    axis.set_ylabel("CCDF (log-y)")
    axis.set_title("Claim count right tail (log-y CCDF)")
    _set_integer_xaxis(axis, int(empirical_x.max()))
    axis.legend()
    fig.tight_layout()
    return fig, axis


def gamma_cdf_diagnostic(
    data: pd.DataFrame,
    axes: tuple[plt.Axes, plt.Axes] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Compare the weighted empirical Severity CDF to Gamma(α̂, β̂) MLE on linear and log-log axes."""
    fig, axes = _cdf_diagnostic_axes(axes, (11.0, 4.5))
    claims = data.loc[data["ClaimNb"] > 0]
    severity = claims["Severity"].to_numpy(dtype=float)
    weights = claims["ClaimNb"].to_numpy(dtype=float)
    alpha, _, beta = gamma.fit(severity, floc=0)
    empirical_x, empirical_f = weighted_empirical_cdf(severity, weights)
    x_grid = np.geomspace(max(severity.min(), 1.0), empirical_x.max() * 1.5, 400)
    theoretical_f = gamma.cdf(x_grid, alpha, scale=beta)
    for axis, log_axes in zip(axes, (False, True)):
        axis.step(
            empirical_x,
            empirical_f,
            where="post",
            color=_EMPIRICAL_COLOR,
            label="Empirical",
        )
        axis.plot(
            x_grid,
            theoretical_f,
            color=_FIT_COLOR,
            label=f"Gamma(α={alpha:.3f}, β={beta:.0f})",
        )
        axis.set_xlabel("Severity (EUR)")
        axis.set_ylabel("CDF")
        if log_axes:
            axis.set_xscale("log")
            axis.set_yscale("log")
            _set_log_y_observed_floor(axis, empirical_f.min())
            axis.set_title("Severity (log-log)")
        else:
            axis.set_title("Severity (linear)")
        axis.legend()
    fig.tight_layout()
    return fig, axes


def gamma_ccdf_diagnostic(
    data: pd.DataFrame,
    claims: pd.DataFrame | None = None,
    axes: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Compare the empirical severity CCDF to Gamma and Generalized-Pareto sf on log-log axes.

    When ``claims`` (the per-claim DataFrame from :func:`prepare_mtpl_data`) is
    supplied, the empirical and both fits switch to per-claim uncapped
    ``ClaimAmount`` so the cap at ``CLAIM_CAP`` does not flatten the tail in
    the diagnostic. With ``claims=None`` the function falls back to the
    per-policy ``Severity`` (capped, claim-weighted) — kept for back
    compatibility and for callers that do not have per-claim data.

    The x-axis is capped at the empirical 99.5th percentile so the single
    most expensive claim does not stretch the panel over multiple decades
    and squash the bulk; both fitted lines are drawn only up to the same
    cap so all three lines share the visible x-range.
    """
    fig, axis = _single_panel_axes(axes, (6.5, 4.5))
    if claims is not None:
        severity = claims["ClaimAmount"].to_numpy(dtype=float)
        weights = np.ones_like(severity)
        empirical_label = "Empirical (uncapped per-claim)"
    else:
        claim_rows = data.loc[data["ClaimNb"] > 0]
        severity = claim_rows["Severity"].to_numpy(dtype=float)
        weights = claim_rows["ClaimNb"].to_numpy(dtype=float)
        empirical_label = "Empirical (per-policy avg, capped)"
    alpha, _, beta = gamma.fit(severity, floc=0)
    c_gp, _, scale_gp = genpareto.fit(severity, floc=0)
    upper = float(np.quantile(severity, 0.995, weights=weights, method="inverted_cdf"))
    empirical_x, empirical_s = weighted_empirical_ccdf(severity, weights)
    within = empirical_x <= upper
    empirical_x = empirical_x[within]
    empirical_s = empirical_s[within]
    x_grid = np.geomspace(max(severity.min(), 1.0), upper, 400)
    theoretical_s = gamma.sf(x_grid, alpha, scale=beta)
    gp_s = genpareto.sf(x_grid, c_gp, loc=0, scale=scale_gp)
    axis.step(
        empirical_x,
        empirical_s,
        where="post",
        color=_EMPIRICAL_COLOR,
        label=empirical_label,
    )
    axis.plot(
        x_grid,
        theoretical_s,
        color=_FIT_COLOR,
        label=f"Gamma(α={alpha:.3f}, β={beta:.0f})",
    )
    axis.plot(
        x_grid,
        gp_s,
        color=_FIT_COLOR_2,
        label=f"GenPareto(c={c_gp:.3f}, scale={scale_gp:.0f})",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    _set_log_y_observed_floor(axis, empirical_s.min())
    _set_upper_x_bound(axis, upper)
    axis.set_xlabel("Severity (EUR, log scale)")
    axis.set_ylabel("CCDF (log scale)")
    axis.set_title("Severity right tail (log-log CCDF)")
    axis.legend()
    fig.tight_layout()
    return fig, axis


def tweedie_cdf_diagnostic(
    data: pd.DataFrame,
    tweedie_power: float,
    axes: tuple[plt.Axes, plt.Axes] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Compare the weighted empirical per-policy ClaimAmountCapped CDF to Tweedie(p, μ̂, φ̂).

    The fit is profile maximum likelihood: scan over μ on the log scale, with φ
    solved from Pearson residuals at each μ, and pick the μ that maximizes
    glum's series-based Tweedie log-likelihood. φ̂ is re-evaluated at μ̂ for
    the legend; matches what ``select_tweedie_power`` does internally so the
    diagnostic is consistent with the rest of the tutorial. Falls back to
    the moment-based estimator when MLE cannot return finite values.

    The Tweedie CDF is computed by truncating the Poisson-Gamma series
    (the standard actuarial approach; ``scipy.stats.tweedie`` is not exposed in
    the locked scipy build).
    """
    fig, axes = _cdf_diagnostic_axes(axes, (11.0, 4.5))
    amounts = data["ClaimAmountCapped"].to_numpy(dtype=float)
    weights = data["Exposure"].to_numpy(dtype=float)
    mu, phi = _fit_tweedie_mle(amounts, tweedie_power, weights)
    empirical_x, empirical_f = weighted_empirical_cdf(amounts, weights)
    x_grid = np.linspace(0.0, max(amounts.max(), 1.0), 400)
    theoretical_f = tweedie_cdf_series(x_grid, mu, phi, tweedie_power)
    for axis, log_y in zip(axes, (False, True)):
        axis.step(
            empirical_x,
            empirical_f,
            where="post",
            color=_EMPIRICAL_COLOR,
            label="Empirical",
        )
        axis.plot(
            x_grid,
            theoretical_f,
            color=_FIT_COLOR,
            label=f"Tweedie(p={tweedie_power:.1f}, μ={mu:.2f}, φ={phi:.2f})",
        )
        axis.set_xlabel("ClaimAmountCapped (EUR)")
        axis.set_ylabel("CDF")
        axis.set_title("Pure premium" + (" (log-y)" if log_y else " (linear)"))
        if log_y:
            axis.set_yscale("log")
            _set_log_y_observed_floor(axis, empirical_f.min(), factor=10.0)
        axis.legend()
    fig.tight_layout()
    return fig, axes


def tweedie_ccdf_diagnostic(
    data: pd.DataFrame,
    tweedie_power: float,
    axes: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Compare the weighted empirical per-policy CCDF to 1 − Tweedie(p, μ̂, φ̂) on log-log axes.

    The empirical uses ``ClaimAmountOriginal`` (per-policy uncapped aggregate,
    exposure-weighted) rather than ``ClaimAmountCapped`` so the cap at
    ``CLAIM_CAP`` does not flatten the tail in the diagnostic. The x-axis is
    cropped to the positive support (1st positive percentile and 99.5th
    percentile of the positive empirical mass) so log-x succeeds and the
    small-mass extreme does not stretch the panel; the zero-mass step the
    linear CDF panel documents is dropped here. The (μ̂, φ̂) fit is profile MLE
    (see :func:`tweedie_cdf_diagnostic` for details); the log-y lower bound
    uses one decade of headroom instead of two because the Tweedie CCDF decays
    faster than the Gamma one on this data and the extra empty space below
    doesn't carry diagnostic signal.
    """
    fig, axis = _single_panel_axes(axes, (6.5, 4.5))
    amounts = data["ClaimAmountOriginal"].to_numpy(dtype=float)
    weights = data["Exposure"].to_numpy(dtype=float)
    mu, phi = _fit_tweedie_mle(amounts, tweedie_power, weights)
    positive = amounts > 0
    positive_x = amounts[positive]
    positive_w = weights[positive]
    quantiles = np.quantile(
        positive_x, [0.01, 0.995], weights=positive_w, method="inverted_cdf"
    )
    lower, upper = float(quantiles[0]), float(quantiles[1])
    empirical_x, empirical_s = weighted_empirical_ccdf(amounts, weights)
    within = (empirical_x >= lower) & (empirical_x <= upper)
    empirical_x = empirical_x[within]
    empirical_s = empirical_s[within]
    x_grid = np.geomspace(lower, upper, 400)
    theoretical_s = tweedie_sf_series(x_grid, mu, phi, tweedie_power)
    axis.step(
        empirical_x,
        empirical_s,
        where="post",
        color=_EMPIRICAL_COLOR,
        label="Empirical (per-policy uncapped)",
    )
    axis.plot(
        x_grid,
        theoretical_s,
        color=_FIT_COLOR,
        label=f"Tweedie(p={tweedie_power:.1f}, μ={mu:.2f}, φ={phi:.2f})",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    _set_log_y_observed_floor(axis, empirical_s.min(), factor=10.0)
    _set_upper_x_bound(axis, upper)
    axis.set_xlabel("ClaimAmountOriginal (EUR, log scale)")
    axis.set_ylabel("CCDF (log scale)")
    axis.set_title("Pure premium right tail (log-log CCDF)")
    axis.legend()
    fig.tight_layout()
    return fig, axis


def tweedie_cdf_series(
    x: np.ndarray,
    mu: float,
    phi: float,
    tweedie_power: float,
    max_terms: int = 200,
    tol: float = 1e-10,
) -> np.ndarray:
    """Compute the compound Poisson-Gamma Tweedie CDF by truncating its series.

    For ``1 < p < 2`` the Tweedie compound distribution has the series form

    .. math::

        F(y) = e^{-\\lambda} \\sum_{k=0}^{\\infty}
            \\frac{\\lambda^k}{k!} G(y; k\\alpha, \\beta),

    with ``alpha = (2-p)/(p-1)``, ``beta = phi * (p-1) * mu ** (p-1)`` and
    ``lambda = mu ** (2-p) / (phi * (2-p))``. The Poisson weights decay
    geometrically past ``k = lambda`` so the series converges in a few dozen
    terms for typical actuarial data; we stop when the largest per-term
    contribution drops below ``tol``.
    """
    if not (1 < tweedie_power < 2):
        raise ValueError("tweedie_power must lie in (1, 2)")
    if mu <= 0 or phi <= 0:
        raise ValueError("mu and phi must be positive")
    alpha = (2 - tweedie_power) / (tweedie_power - 1)
    beta = phi * (tweedie_power - 1) * mu ** (tweedie_power - 1)
    lam = mu ** (2 - tweedie_power) / (phi * (2 - tweedie_power))
    zero_mass = float(np.exp(-lam))
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    result = np.where(x_arr <= 0, 0.0, zero_mass)
    positive = x_arr > 0
    if not np.any(positive):
        return result
    poisson_term = zero_mass
    for k in range(1, max_terms + 1):
        poisson_term *= lam / k
        contribution = poisson_term * gamma.cdf(
            np.maximum(x_arr[positive], 1e-300), a=k * alpha, scale=beta
        )
        result = result.copy()
        result[positive] += contribution
        if float(np.max(contribution)) < tol:
            break
    return result


def tweedie_sf_series(
    x: np.ndarray,
    mu: float,
    phi: float,
    tweedie_power: float,
    max_terms: int = 200,
    tol: float = 1e-10,
) -> np.ndarray:
    """Compute the compound Poisson-Gamma Tweedie survival function 1 − F by series truncation.

    Same convergence guarantee as ``tweedie_cdf_series``; computes 1 − F by
    subtracting ``gamma.cdf`` terms from ``1 − zero_mass``. Mathematically
    equivalent to summing ``poisson_term · gamma.sf``, kept as ``1 − ...``
    so the loop shares its structure with ``tweedie_cdf_series``. The trailing
    ``np.maximum(result, 0.0)`` clamps sub-epsilon cancellation noise.
    """
    if not (1 < tweedie_power < 2):
        raise ValueError("tweedie_power must lie in (1, 2)")
    if mu <= 0 or phi <= 0:
        raise ValueError("mu and phi must be positive")
    alpha = (2 - tweedie_power) / (tweedie_power - 1)
    beta = phi * (tweedie_power - 1) * mu ** (tweedie_power - 1)
    lam = mu ** (2 - tweedie_power) / (phi * (2 - tweedie_power))
    zero_mass = float(np.exp(-lam))
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    result = np.where(x_arr <= 0, 1.0, 1.0 - zero_mass)
    positive = x_arr > 0
    if not np.any(positive):
        return result
    poisson_term = zero_mass
    for k in range(1, max_terms + 1):
        poisson_term *= lam / k
        contribution = poisson_term * gamma.cdf(
            np.maximum(x_arr[positive], 1e-300), a=k * alpha, scale=beta
        )
        result = result.copy()
        result[positive] -= contribution
        if float(np.max(contribution)) < tol:
            break
    return np.maximum(result, 0.0)


_INITIAL_VARIANCE_POWER = {"frequency": 1.0, "severity": 2.0, "pure_premium": 1.5}


def mean_variance_table(
    data: pd.DataFrame, by: str, response: str
) -> tuple[pd.DataFrame, float, float]:
    """Group by ``by``, compute the weighted mean and variance of ``response`` per group.

    Returns the per-group table plus the fitted ``(p, φ)`` of the power-law
    variance function ``Var ≈ φ · mean^p``. The fit uses ``scipy.optimize.curve_fit``
    starting from the canonical power for the chosen response.
    """
    rows, target, weight = target_and_weight(data, response)
    frame = rows.assign(target=target, weight=weight).rename(columns={by: "_group"})
    grouped = frame.groupby("_group", observed=True)

    def _aggregate(group: pd.DataFrame) -> pd.Series:
        weights = group["weight"].to_numpy(dtype=float)
        target = group["target"].to_numpy(dtype=float)
        mean = float(np.average(target, weights=weights))
        variance = float(np.average((target - mean) ** 2, weights=weights))
        return pd.Series({"n": len(group), "mean": mean, "variance": variance})

    table = grouped.apply(_aggregate, include_groups=False).reset_index()
    mean_values = table["mean"].to_numpy(dtype=float)
    variance_values = table["variance"].to_numpy(dtype=float)
    initial_p = _INITIAL_VARIANCE_POWER[response]
    initial_phi = (
        float(np.mean(variance_values) / np.mean(mean_values) ** initial_p)
        if np.all(mean_values > 0)
        else 1.0
    )
    (p_fit, phi_fit), _ = curve_fit(
        lambda mean, p, phi: phi * np.maximum(mean, 1e-12) ** p,
        mean_values,
        variance_values,
        p0=[initial_p, initial_phi],
    )
    return table, float(p_fit), float(phi_fit)


def mean_variance_diagnostic(
    data: pd.DataFrame,
    by: str,
    tweedie_power: float,
    axes: tuple[plt.Axes, plt.Axes, plt.Axes] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """Plot empirical (μ, σ²) per group and the fitted power-law variance function.

    The pure-premium panel uses ``tweedie_power`` only as the legend label and the
    starting point of the curve fit; the fitted ``p`` is what the data says.
    """
    responses = ("frequency", "severity", "pure_premium")
    titles = {
        "frequency": "Frequency",
        "severity": "Severity",
        "pure_premium": "Pure premium",
    }
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    else:
        fig = axes[0].figure
    for axis, response in zip(axes, responses):
        table, p, phi = mean_variance_table(data, by, response)
        axis.scatter(
            table["mean"],
            table["variance"],
            color="black",
            label=f"Empirical ({len(table)} groups)",
        )
        positive_mean = table["mean"][table["mean"] > 0]
        x_grid = np.linspace(
            max(positive_mean.min(), 1e-12),
            positive_mean.max() * 1.5,
            100,
        )
        axis.plot(
            x_grid,
            phi * x_grid**p,
            "--",
            label=f"Fit: Var ≈ {phi:.3g} · μ^{p:.2f}",
        )
        axis.set_xlabel(f"Mean of {titles[response]}")
        axis.set_ylabel(f"Variance of {titles[response]}")
        title = f"{titles[response]} by {by}"
        if response == "pure_premium":
            title = f"{title} (Tweedie p = {tweedie_power:.1f})"
        axis.set_title(title)
        axis.legend()
    fig.tight_layout()
    return fig, axes


def glum_family(component: str, tweedie_power: float) -> str | TweedieDistribution:
    """Map tutorial components to GLUM families."""
    if component == "pure_premium":
        return TweedieDistribution(tweedie_power)
    return TARGET_SPECS[component].family


def fit_glum_predictive(
    data: pd.DataFrame,
    component: str,
    tweedie_power: float,
    n_alphas: int = 30,
) -> GeneralizedLinearRegressorCV:
    """Fit the penalized GLUM model used for prediction."""
    rows, target, weight = target_and_weight(data, component)
    if component == "severity":
        anchor = float(np.average(target, weights=weight))
        anchor_indices = {
            data.index[data[column] == category][0]
            for column in CATEGORICAL_FEATURES
            for category in data[column].unique()
            if category not in set(rows[column].unique())
        }
        if anchor_indices:
            rows = pd.concat([rows, data.loc[list(anchor_indices)]])
            target = np.r_[target, np.full(len(anchor_indices), anchor)]
            weight = np.r_[weight, np.zeros(len(anchor_indices))]
    model = GeneralizedLinearRegressorCV(
        family=glum_family(component, tweedie_power),
        link="log",
        formula=RATING_FORMULA,
        drop_first=True,
        cv=3,
        n_alphas=n_alphas,
        min_alpha_ratio=1e-3,
        l1_ratio=[0.0, 0.5, 1.0],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_iter=200,
        gradient_tol=5e-4,
        scale_predictors=True,
    )
    return model.fit(rows, target, sample_weight=weight)


def fit_glum_inference(
    data: pd.DataFrame,
    component: str,
    tweedie_power: float,
) -> GeneralizedLinearRegressor:
    """Fit a separate unregularized GLUM for classical robust inference."""
    rows, target, weight = target_and_weight(data, component)
    rows = rows.copy()
    for column in CATEGORICAL_FEATURES:
        rows[column] = rows[column].cat.remove_unused_categories()
    model = GeneralizedLinearRegressor(
        family=glum_family(component, tweedie_power),
        link="log",
        formula=RATING_FORMULA,
        drop_first=True,
        alpha=0,
        robust=True,
        max_iter=200,
    )
    return model.fit(
        rows,
        target,
        sample_weight=weight,
        store_covariance_matrix=True,
    )


def coefficient_relativity_table(
    model: GeneralizedLinearRegressor,
) -> pd.DataFrame:
    """Return compact classical coefficient and pricing-relativity output."""
    table = model.coef_table().rename_axis("term").reset_index()
    table = table[["term", "coef", "se", "p_value", "ci_lower", "ci_upper"]].rename(
        columns={"se": "std_error"}
    )
    table["relativity"] = np.exp(table["coef"].clip(-700, 700))
    table["relativity_lower"] = np.exp(table["ci_lower"].clip(-700, 700))
    table["relativity_upper"] = np.exp(table["ci_upper"].clip(-700, 700))
    return table


def select_tweedie_power(
    development_train: pd.DataFrame,
    development_validation: pd.DataFrame,
    powers: tuple[float, ...] = TWEEDIE_POWERS,
) -> tuple[float, pd.DataFrame]:
    """Select p by validation profile likelihood under the fitted rate convention."""
    train_rows, y_train, w_train = target_and_weight(development_train, "pure_premium")
    validation_rows, y_validation, w_validation = target_and_weight(
        development_validation, "pure_premium"
    )
    results = []
    for power in powers:
        family = TweedieDistribution(power)
        model = GeneralizedLinearRegressor(
            family=family,
            link="log",
            formula=RATING_FORMULA,
            drop_first=True,
            alpha=0,
            max_iter=200,
        ).fit(train_rows, y_train, sample_weight=w_train)
        train_prediction = model.predict(train_rows)
        validation_prediction = model.predict(validation_rows)
        dispersion = family.dispersion(
            y_train,
            train_prediction,
            sample_weight=w_train,
            ddof=len(model.coef_) + 1,
        )
        log_likelihood = family.log_likelihood(
            y_validation,
            validation_prediction,
            sample_weight=w_validation,
            dispersion=dispersion,
        )
        results.append(
            {
                "power": power,
                "training_dispersion": dispersion,
                "validation_log_likelihood": log_likelihood,
            }
        )
    profile = pd.DataFrame(results)
    if not np.isfinite(profile["validation_log_likelihood"]).all():
        raise RuntimeError("Tweedie profile likelihood produced non-finite values")
    selected = float(
        profile.loc[profile["validation_log_likelihood"].idxmax(), "power"]
    )
    return selected, profile


def lightgbm_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return the native numeric/categorical LightGBM feature frame."""
    return data[LIGHTGBM_FEATURES]


def lightgbm_params(
    component: str,
    tweedie_power: float,
    num_leaves: int,
    min_child_samples: int,
) -> dict[str, Any]:
    """Return the fixed LightGBM settings plus one compact grid point."""
    objective = TARGET_SPECS[component].family
    params: dict[str, Any] = {
        "objective": objective,
        "metric": objective,
        "learning_rate": LEARNING_RATE,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "lambda_l2": REG_LAMBDA,
        "seed": RANDOM_STATE,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    if component == "pure_premium":
        params["tweedie_variance_power"] = tweedie_power
    return params


def weighted_mean_deviance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weight: np.ndarray,
    component: str,
    tweedie_power: float,
) -> float:
    """Compute the component-appropriate weighted mean deviance."""
    if component == "frequency":
        return float(mean_poisson_deviance(y_true, y_pred, sample_weight=weight))
    if component == "severity":
        return float(mean_gamma_deviance(y_true, y_pred, sample_weight=weight))
    return float(
        mean_tweedie_deviance(
            y_true,
            y_pred,
            sample_weight=weight,
            power=tweedie_power,
        )
    )


def fit_lightgbm(
    development_train: pd.DataFrame,
    development_validation: pd.DataFrame,
    full_train: pd.DataFrame,
    component: str,
    tweedie_power: float,
    max_rounds: int = NUM_BOOST_ROUND,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> tuple[lgb.Booster, pd.DataFrame]:
    """Tune on development data, then refit all training rows for best_iteration."""
    train_rows, y_train, w_train = target_and_weight(development_train, component)
    validation_rows, y_validation, w_validation = target_and_weight(
        development_validation, component
    )
    min_child_grid = (20, 50) if component == "severity" else (100, 500)
    tuning_rows: list[dict[str, int | float]] = []

    for num_leaves in (15, 31):
        for min_child_samples in min_child_grid:
            params = lightgbm_params(
                component,
                tweedie_power,
                num_leaves,
                min_child_samples,
            )
            train_set = lgb.Dataset(
                lightgbm_frame(train_rows),
                label=y_train,
                weight=w_train,
                categorical_feature=CATEGORICAL_FEATURES,
            )
            validation_set = lgb.Dataset(
                lightgbm_frame(validation_rows),
                label=y_validation,
                weight=w_validation,
                categorical_feature=CATEGORICAL_FEATURES,
                reference=train_set,
            )
            tuned = lgb.train(
                params,
                train_set,
                num_boost_round=max_rounds,
                valid_sets=[validation_set],
                valid_names=["validation"],
                callbacks=[
                    lgb.early_stopping(
                        early_stopping_rounds,
                        first_metric_only=True,
                        verbose=False,
                    )
                ],
            )
            prediction = np.asarray(
                tuned.predict(
                    lightgbm_frame(validation_rows),
                    num_iteration=tuned.best_iteration,
                ),
                dtype=float,
            )
            tuning_rows.append(
                {
                    "num_leaves": num_leaves,
                    "min_child_samples": min_child_samples,
                    "best_iteration": tuned.best_iteration,
                    "validation_deviance": weighted_mean_deviance(
                        y_validation,
                        prediction,
                        w_validation,
                        component,
                        tweedie_power,
                    ),
                }
            )

    tuning = pd.DataFrame(tuning_rows).sort_values(
        ["validation_deviance", "num_leaves", "min_child_samples"],
        ignore_index=True,
    )
    best = tuning.iloc[0]
    full_rows, y_full, w_full = target_and_weight(full_train, component)
    full_set = lgb.Dataset(
        lightgbm_frame(full_rows),
        label=y_full,
        weight=w_full,
        categorical_feature=CATEGORICAL_FEATURES,
    )
    final = lgb.train(
        lightgbm_params(
            component,
            tweedie_power,
            int(best["num_leaves"]),
            int(best["min_child_samples"]),
        ),
        full_set,
        num_boost_round=int(best["best_iteration"]),
    )
    return final, tuning


def lorenz_curve(
    y_true_rate: np.ndarray,
    y_pred_rate: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a low-to-high Lorenz curve including both endpoints.

    Tied predictions are aggregated into a single block so the curve is
    permutation-invariant. The block midpoint is reached once, giving a
    straight line across the tied block (the standard convention when
    average ranks are used).
    """
    y_true_rate = np.asarray(y_true_rate, dtype=float)
    y_pred_rate = np.asarray(y_pred_rate, dtype=float)
    weight = np.asarray(weight, dtype=float)
    if np.any(weight <= 0) or weight.sum() <= 0:
        raise ValueError("Lorenz weights must be positive")
    order = np.argsort(y_pred_rate, kind="stable")
    sorted_pred = y_pred_rate[order]
    sorted_weight = weight[order]
    sorted_observed = y_true_rate[order]
    _, starts = np.unique(sorted_pred, return_index=True)
    block_weight = np.add.reduceat(sorted_weight, starts)
    block_observed = np.add.reduceat(sorted_weight * sorted_observed, starts)
    cumulative_weight = np.r_[0.0, np.cumsum(block_weight)]
    cumulative_loss = np.r_[0.0, np.cumsum(block_observed)]
    cumulative_weight /= cumulative_weight[-1]
    if cumulative_loss[-1] > 0:
        cumulative_loss /= cumulative_loss[-1]
    return cumulative_weight, cumulative_loss


def portfolio_calibration_table(
    data: pd.DataFrame,
    component: str,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compare observed and predicted portfolio totals for one response."""
    _, target, weight = target_and_weight(data, component)
    observed_total = float(np.sum(target * weight))
    records = []
    for model, prediction in predictions.items():
        predicted_total = float(np.sum(np.asarray(prediction, dtype=float) * weight))
        records.append(
            {
                "model": model,
                "observed_total": observed_total,
                "predicted_total": predicted_total,
                "actual_expected": observed_total / predicted_total,
                "signed_percentage_error": 100
                * (predicted_total - observed_total)
                / observed_total,
            }
        )
    return pd.DataFrame(records).set_index("model")


def lorenz_dominance_table(
    y_true_rate: np.ndarray,
    predictions: dict[str, np.ndarray],
    weight: np.ndarray,
) -> pd.DataFrame:
    """Compare low-to-high Lorenz curves; a lower curve dominates."""
    curves = {
        name: lorenz_curve(y_true_rate, prediction, weight)
        for name, prediction in predictions.items()
    }
    rows = []
    for model_a, model_b in combinations(predictions, 2):
        x = np.unique(np.r_[curves[model_a][0], curves[model_b][0]])
        a = np.interp(x, *curves[model_a])
        b = np.interp(x, *curves[model_b])
        if np.all(a <= b) and np.any(a < b):
            result = f"{model_a} dominates {model_b}"
        elif np.all(b <= a) and np.any(b < a):
            result = f"{model_b} dominates {model_a}"
        elif np.any(a < b) and np.any(a > b):
            result = "crossing; no dominance"
        else:
            result = "tied; no dominance"
        rows.append({"model_a": model_a, "model_b": model_b, "result": result})
    return pd.DataFrame(rows)


def gini(
    y_true_rate: np.ndarray,
    y_pred_rate: np.ndarray,
    weight: np.ndarray,
) -> float:
    """Compute the weighted Gini coefficient for a low-to-high ordering.

    Uses the average-rank (midrank) closed form

        G = 2 * sum_i (w_i y_i Fbar_i) / sum_i (w_i y_i) - 1,
        Fbar_i = (W_{i-1} + w_i / 2) / W,

    which is the Frees-Meyers-Cummings convention. Tied predictions are
    aggregated into single blocks so the value is permutation-invariant;
    the resulting Gini equals 1 - 2 * AUC of the tie-corrected polygonal
    Lorenz curve. Returns ``numpy.nan`` when total weighted loss is not
    positive (no average can be defined).
    """
    y_true_rate = np.asarray(y_true_rate, dtype=float)
    y_pred_rate = np.asarray(y_pred_rate, dtype=float)
    weight = np.asarray(weight, dtype=float)
    if np.any(weight <= 0) or weight.sum() <= 0:
        raise ValueError("Gini weights must be positive")
    order = np.argsort(y_pred_rate, kind="stable")
    sorted_pred = y_pred_rate[order]
    sorted_weight = weight[order]
    sorted_observed = y_true_rate[order]
    _, starts = np.unique(sorted_pred, return_index=True)
    block_weight = np.add.reduceat(sorted_weight, starts)
    block_weighted_y = np.add.reduceat(sorted_weight * sorted_observed, starts)
    total_weight = block_weight.sum()
    total_loss = block_weighted_y.sum()
    if total_loss <= 0:
        return float("nan")
    cumulative_weight = np.cumsum(block_weight)
    f_mid = (cumulative_weight - block_weight / 2) / total_weight
    return float(2.0 * (block_weighted_y * f_mid).sum() / total_loss - 1.0)


def exposure_balanced_lift_table(
    y_true_rate: np.ndarray,
    y_pred_rate: np.ndarray,
    exposure: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Aggregate lift at exactly equal exposure boundaries, splitting boundary rows."""
    y_true_rate = np.asarray(y_true_rate, dtype=float)
    y_pred_rate = np.asarray(y_pred_rate, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    if n_bins < 1 or np.any(exposure <= 0):
        raise ValueError("n_bins and exposure must be positive")

    order = np.argsort(y_pred_rate, kind="stable")
    cumulative_exposure = np.r_[0.0, np.cumsum(exposure[order])]
    cumulative_observed = np.r_[0.0, np.cumsum(exposure[order] * y_true_rate[order])]
    cumulative_predicted = np.r_[0.0, np.cumsum(exposure[order] * y_pred_rate[order])]
    boundaries = np.linspace(0, cumulative_exposure[-1], n_bins + 1)
    observed = np.interp(boundaries, cumulative_exposure, cumulative_observed)
    predicted = np.interp(boundaries, cumulative_exposure, cumulative_predicted)
    bin_exposure = np.diff(boundaries)
    return pd.DataFrame(
        {
            "exposure": bin_exposure,
            "observed_rate": np.diff(observed) / bin_exposure,
            "predicted_rate": np.diff(predicted) / bin_exposure,
        },
        index=pd.RangeIndex(1, n_bins + 1, name="decile"),
    )


def exposure_balanced_double_lift_table(
    y_true_rate: np.ndarray,
    model_a_prediction: np.ndarray,
    model_b_prediction: np.ndarray,
    exposure: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin observed rates by the low-to-high model-A/model-B prediction ratio."""
    y_true_rate = np.asarray(y_true_rate, dtype=float)
    model_a_prediction = np.asarray(model_a_prediction, dtype=float)
    model_b_prediction = np.asarray(model_b_prediction, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if not np.all(np.isfinite(exposure) & (exposure > 0)):
        raise ValueError("exposure must be finite and positive")
    if not np.all(np.isfinite(model_a_prediction) & (model_a_prediction > 0)):
        raise ValueError("model_a_prediction must be finite and positive")
    if not np.all(np.isfinite(model_b_prediction) & (model_b_prediction > 0)):
        raise ValueError("model_b_prediction must be finite and positive")
    ratio = model_a_prediction / model_b_prediction
    order = np.argsort(ratio, kind="stable")
    cumulative_exposure = np.r_[0.0, np.cumsum(exposure[order])]
    boundaries = np.linspace(0, cumulative_exposure[-1], n_bins + 1)
    bin_exposure = np.diff(boundaries)

    def rates(values: np.ndarray) -> np.ndarray:
        cumulative = np.r_[0.0, np.cumsum(exposure[order] * values[order])]
        return (
            np.diff(np.interp(boundaries, cumulative_exposure, cumulative))
            / bin_exposure
        )

    return pd.DataFrame(
        {
            "exposure": bin_exposure,
            "observed_rate": rates(y_true_rate),
            "model_a_rate": rates(model_a_prediction),
            "model_b_rate": rates(model_b_prediction),
            "prediction_ratio": rates(ratio),
        },
        index=pd.RangeIndex(1, n_bins + 1, name="decile"),
    )


def lift_diagnostics(
    y_true_rate: np.ndarray,
    predictions: dict[str, np.ndarray],
    exposure: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Report safest-to-riskiest observed and predicted lift ranges."""
    rows = []
    for model, prediction in predictions.items():
        lift = exposure_balanced_lift_table(y_true_rate, prediction, exposure, n_bins)
        rows.append(
            {
                "model": model,
                "observed_first": lift["observed_rate"].iloc[0],
                "observed_last": lift["observed_rate"].iloc[-1],
                "observed_range": lift["observed_rate"].iloc[-1]
                - lift["observed_rate"].iloc[0],
                "predicted_first": lift["predicted_rate"].iloc[0],
                "predicted_last": lift["predicted_rate"].iloc[-1],
                "predicted_range": lift["predicted_rate"].iloc[-1]
                - lift["predicted_rate"].iloc[0],
            }
        )
    return pd.DataFrame(rows).set_index("model")


def evaluate_predictions(
    data: pd.DataFrame,
    component: str,
    predictions: dict[str, np.ndarray],
    tweedie_power: float,
) -> pd.DataFrame:
    """Report weighted deviance, D2, A/E, raw Gini, and normalized Gini."""
    rows, target, weight = target_and_weight(data, component)
    null_prediction = np.full_like(target, np.average(target, weights=weight))
    null_deviance = weighted_mean_deviance(
        target, null_prediction, weight, component, tweedie_power
    )
    perfect_gini = gini(target, target, weight)
    results = []
    for name, prediction in predictions.items():
        prediction = np.asarray(prediction, dtype=float)
        if len(prediction) != len(rows):
            raise ValueError(f"{name} prediction length does not match evaluation rows")
        if not np.all(np.isfinite(prediction) & (prediction > 0)):
            raise ValueError(f"{name} produced non-finite or non-positive predictions")
        deviance = weighted_mean_deviance(
            target, prediction, weight, component, tweedie_power
        )
        raw_gini = gini(target, prediction, weight)
        results.append(
            {
                "model": name,
                "weighted_deviance": deviance,
                "D2": 1 - deviance / null_deviance,
                "actual_expected": np.sum(weight * target)
                / np.sum(weight * prediction),
                "raw_gini": raw_gini,
                "normalized_gini": (
                    raw_gini / perfect_gini if perfect_gini != 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(results).set_index("model")


def grouped_calibration(
    data: pd.DataFrame,
    component: str,
    predictions: dict[str, np.ndarray],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Aggregate observed and predicted component rates by driver-age bands."""
    rows, target, weight = target_and_weight(data, component)
    bands = pd.qcut(rows["DrivAge"], n_bins, duplicates="drop")
    frame = pd.DataFrame(
        {"band": bands.astype(str), "weight": weight, "observed": target * weight}
    )
    for name, prediction in predictions.items():
        frame[name] = np.asarray(prediction) * weight
    grouped = frame.groupby("band", sort=False, observed=True).sum()
    rate_columns = ["observed", *predictions]
    grouped[rate_columns] = grouped[rate_columns].div(grouped["weight"], axis=0)
    return grouped


def save_core_figures(
    test: pd.DataFrame,
    frequency_predictions: dict[str, np.ndarray],
    severity_predictions: dict[str, np.ndarray],
    pure_premium_predictions: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """Save the three core tutorial figures."""
    expected = {
        "GLUM frequency x severity",
        "LightGBM frequency x severity",
        "GLUM Tweedie",
        "LightGBM Tweedie",
    }
    if set(pure_premium_predictions) != expected:
        raise ValueError(
            f"pure_premium_predictions must contain exactly {sorted(expected)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    with plt.style.context("fivethirtyeight"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for axis, component, predictions in (
            (axes[0], "frequency", frequency_predictions),
            (axes[1], "severity", severity_predictions),
        ):
            calibration = grouped_calibration(test, component, predictions)
            x = np.arange(len(calibration))
            axis.plot(x, calibration["observed"], "o-", label="Observed")
            for name in predictions:
                axis.plot(x, calibration[name], "--", label=name)
            axis.set(
                title=f"{component.replace('_', ' ').title()} by driver age",
                xlabel="Driver-age band",
                ylabel="Rate",
            )
            axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "component_calibration.png", dpi=150)
        plt.close(fig)

        y_test = test["PurePremium"].to_numpy(dtype=float)
        exposure = test["Exposure"].to_numpy(dtype=float)
        fig, axis = plt.subplots(figsize=(7, 6))
        axis.plot([0, 1], [0, 1], color="black", linestyle=":", label="Equality")
        for name, prediction in pure_premium_predictions.items():
            x, y = lorenz_curve(y_test, prediction, exposure)
            axis.plot(
                x,
                y,
                label=f"{name} (Gini {gini(y_test, prediction, exposure):.3f})",
            )
        axis.set(
            title="Pure-premium Lorenz curves (test)",
            xlabel="Cumulative exposure",
            ylabel="Cumulative capped loss",
        )
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "pure_premium_lorenz.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
        for axis, (name, prediction) in zip(
            axes.flat, pure_premium_predictions.items(), strict=True
        ):
            lift = exposure_balanced_lift_table(y_test, prediction, exposure)
            axis.plot(lift.index, lift["observed_rate"], "o-", label="Observed")
            axis.plot(lift.index, lift["predicted_rate"], "s--", label="Predicted")
            axis.set(
                title=name,
                xlabel="Exposure-balanced decile",
                ylabel="Pure premium",
            )
            axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "pure_premium_lift.png", dpi=150)
        plt.close(fig)

        comparisons = (
            (
                "GLUM decomposition",
                "GLUM frequency x severity",
                "LightGBM decomposition",
                "LightGBM frequency x severity",
            ),
            (
                "GLUM Tweedie",
                "GLUM Tweedie",
                "LightGBM Tweedie",
                "LightGBM Tweedie",
            ),
            (
                "GLUM decomposition",
                "GLUM frequency x severity",
                "GLUM Tweedie",
                "GLUM Tweedie",
            ),
            (
                "LightGBM decomposition",
                "LightGBM frequency x severity",
                "LightGBM Tweedie",
                "LightGBM Tweedie",
            ),
        )
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
        for axis, (a_label, a_name, b_label, b_name) in zip(
            axes.flat, comparisons, strict=True
        ):
            lift = exposure_balanced_double_lift_table(
                y_test,
                pure_premium_predictions[a_name],
                pure_premium_predictions[b_name],
                exposure,
            )
            axis.plot(lift.index, lift["observed_rate"], "o-", label="Observed")
            axis.plot(lift.index, lift["model_a_rate"], "s--", label=a_label)
            axis.plot(lift.index, lift["model_b_rate"], "^:", label=b_label)
            axis.set(
                title=f"{a_label} / {b_label}",
                xlabel="A/B ratio decile",
                ylabel="Pure premium",
            )
            axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "pure_premium_double_lift.png", dpi=150)
        plt.close(fig)


def hexbin_grid(
    xy_by_model,
    xlabel,
    ylabel,
    logx=False,
    logy=False,
    reference="diag",
    extent_mode="auto",
):
    """Draw one hexbin panel per model on a shared log-count colour scale.

    ``extent_mode`` clips hexbinning (and the axis limits) to the predicted-value
    span when the panel is linear:
    - ``"diag"`` uses ``(xmin, xmax)`` on both axes (predicted vs observed).
    - ``"residual"`` uses ``(xmin, xmax)`` on x and the symmetric predicted span
      on y (residual panel).
    - ``"auto"`` leaves hexbinning and limits alone (log panels).
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    artists = []
    for axis, (name, (x, y)) in zip(axes.flat, xy_by_model.items()):
        xlo, xhi = float(x.min()), float(x.max())
        if extent_mode == "diag" and not logx and not logy:
            ylo, yhi = xlo, xhi
            extent = (xlo, xhi, ylo, yhi)
        elif extent_mode == "residual" and not logx and not logy:
            span = xhi - xlo
            extent = (xlo, xhi, -span, span)
        else:
            extent = None
        hb = axis.hexbin(
            x,
            y,
            gridsize=40,
            mincnt=1,
            cmap="cividis",
            linewidths=0.2,
            xscale="log" if logx else "linear",
            yscale="log" if logy else "linear",
            extent=extent,
        )
        artists.append(hb)
        axis.set_title(name)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        if reference == "diag":
            if extent is not None:
                ref_lo, ref_hi = max(xlo, extent[2]), min(xhi, extent[3])
            else:
                ref_lo, ref_hi = min(x.min(), y.min()), max(x.max(), y.max())
            axis.plot([ref_lo, ref_hi], [ref_lo, ref_hi], ":", color="grey")
        elif reference == "zero":
            axis.axhline(0.0, ls=":", color="grey")
        if extent is not None:
            axis.set_xlim(extent[0], extent[1])
            axis.set_ylim(extent[2], extent[3])
    vmax = max(float(artist.get_array().max()) for artist in artists)
    for artist in artists:
        artist.set_norm(mcolors.LogNorm(1.0, vmax))
    colorbar = fig.colorbar(artists[0], ax=axes, fraction=0.03)
    colorbar.set_label("policies per bin")
    plt.show()


def print_frame(title: str, frame: pd.DataFrame) -> None:
    """Render every tutorial table through the shared Rich console."""
    shown = frame.reset_index() if frame.index.name is not None else frame
    table = Table(title=title, show_lines=False)
    for column in shown.columns:
        table.add_column(
            str(column),
            justify="right" if pd.api.types.is_numeric_dtype(shown[column]) else "left",
        )
    for row in shown.itertuples(index=False, name=None):
        table.add_row(
            *[
                "—"
                if pd.isna(value)
                else f"{value:,.6g}"
                if isinstance(value, (float, np.floating))
                else str(value)
                for value in row
            ]
        )
    CONSOLE.print(table)


def main(
    n_samples: int | None = None,
    n_alphas: int = 30,
    max_rounds: int = NUM_BOOST_ROUND,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    output_dir: Path = Path("artifacts/pure_premium"),
    data: pd.DataFrame | None = None,
    preparation: dict[str, int | float] | None = None,
) -> None:
    """Run the complete tutorial; importing this module performs no work."""
    if data is None:
        data, _, preparation = load_mtpl2(n_samples=n_samples)
    if preparation is None:
        raise ValueError("preparation is required when data is supplied")
    print_frame("Target and weight definitions", target_weight_table())
    print_frame(
        "Claim preparation",
        pd.DataFrame([preparation]).assign(
            capped_loss_share=lambda frame: frame["capped_loss_share"] * 100
        ),
    )

    splits = make_splits(data)
    selected_power, profile = select_tweedie_power(
        splits.development_train,
        splits.development_validation,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output_dir / "tweedie_power_profile.csv", index=False)
    print_frame("Tweedie power profile", profile)
    print(f"\nSelected Tweedie power: {selected_power:.1f}")

    glum_models: dict[str, GeneralizedLinearRegressorCV] = {}
    for component in TARGET_SPECS:
        predictive = fit_glum_predictive(
            splits.train,
            component,
            selected_power,
            n_alphas=n_alphas,
        )
        inference = fit_glum_inference(
            splits.train,
            component,
            selected_power,
        )
        glum_models[component] = predictive
        relativities = coefficient_relativity_table(inference)
        print_frame(f"GLUM {component} relativities", relativities)
        relativities.to_csv(
            output_dir / f"glum_{component}_relativities.csv",
            index=False,
        )
        print(
            f"GLUM {component}: alpha={predictive.alpha_:.6g}, "
            f"l1_ratio={predictive.l1_ratio_:.1f}"
        )
    print(
        "Coefficient intervals are classical robust inference conditional on the "
        "chosen specification; they do not describe the penalized predictive fits."
    )

    lightgbm_models: dict[str, lgb.Booster] = {}
    for component in TARGET_SPECS:
        model, tuning = fit_lightgbm(
            splits.development_train,
            splits.development_validation,
            splits.train,
            component,
            selected_power,
            max_rounds=max_rounds,
            early_stopping_rounds=early_stopping_rounds,
        )
        lightgbm_models[component] = model
        tuning.to_csv(output_dir / f"lightgbm_{component}_tuning.csv", index=False)
        print_frame(f"LightGBM {component} tuning", tuning)

    test_features = lightgbm_frame(splits.test)
    glum_frequency = np.asarray(
        glum_models["frequency"].predict(splits.test), dtype=float
    )
    glum_severity = np.asarray(
        glum_models["severity"].predict(splits.test), dtype=float
    )
    lgb_frequency = np.asarray(
        lightgbm_models["frequency"].predict(test_features), dtype=float
    )
    lgb_severity = np.asarray(
        lightgbm_models["severity"].predict(test_features), dtype=float
    )
    frequency_predictions = {
        "GLUM": glum_frequency,
        "LightGBM": lgb_frequency,
    }
    severity_rows, _, _ = target_and_weight(splits.test, "severity")
    severity_features = lightgbm_frame(severity_rows)
    severity_predictions = {
        "GLUM": np.asarray(glum_models["severity"].predict(severity_rows), dtype=float),
        "LightGBM": np.asarray(
            lightgbm_models["severity"].predict(severity_features), dtype=float
        ),
    }
    pure_premium_predictions = {
        "GLUM frequency x severity": glum_frequency * glum_severity,
        "LightGBM frequency x severity": lgb_frequency * lgb_severity,
        "GLUM Tweedie": np.asarray(
            glum_models["pure_premium"].predict(splits.test), dtype=float
        ),
        "LightGBM Tweedie": np.asarray(
            lightgbm_models["pure_premium"].predict(test_features), dtype=float
        ),
    }

    print_frame(
        "Frequency validation",
        evaluate_predictions(
            splits.test,
            "frequency",
            frequency_predictions,
            selected_power,
        ),
    )
    print_frame(
        "Severity validation",
        evaluate_predictions(
            splits.test,
            "severity",
            severity_predictions,
            selected_power,
        ),
    )
    print_frame(
        "Pure-premium validation",
        evaluate_predictions(
            splits.test,
            "pure_premium",
            pure_premium_predictions,
            selected_power,
        ),
    )
    y_test = splits.test["PurePremium"].to_numpy(dtype=float)
    exposure = splits.test["Exposure"].to_numpy(dtype=float)
    for component, predictions in (
        ("frequency", frequency_predictions),
        ("severity", severity_predictions),
        ("pure_premium", pure_premium_predictions),
    ):
        print_frame(
            f"{component.replace('_', ' ').title()} portfolio calibration",
            portfolio_calibration_table(splits.test, component, predictions),
        )
    print_frame(
        "Pure-premium Lorenz dominance",
        lorenz_dominance_table(y_test, pure_premium_predictions, exposure),
    )
    print_frame(
        "Pure-premium lift ranges",
        lift_diagnostics(y_test, pure_premium_predictions, exposure),
    )
    for model_a, model_b in combinations(pure_premium_predictions, 2):
        print_frame(
            f"Double lift: {model_a} / {model_b}",
            exposure_balanced_double_lift_table(
                y_test,
                pure_premium_predictions[model_a],
                pure_premium_predictions[model_b],
                exposure,
            ),
        )
    save_core_figures(
        splits.test,
        frequency_predictions,
        severity_predictions,
        pure_premium_predictions,
        output_dir,
    )
    print(f"\nSaved tables and core figures to {output_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the small set of controls useful for tutorial smoke runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--n-alphas", type=int, default=30)
    parser.add_argument("--max-rounds", type=int, default=NUM_BOOST_ROUND)
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=EARLY_STOPPING_ROUNDS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/pure_premium"),
    )
    return parser.parse_args(argv)


def cli(argv: list[str] | None = None) -> None:
    """Parse command-line options and run the tutorial."""
    arguments = parse_args(argv)
    main(
        n_samples=arguments.n_samples,
        n_alphas=arguments.n_alphas,
        max_rounds=arguments.max_rounds,
        early_stopping_rounds=arguments.early_stopping_rounds,
        output_dir=arguments.output_dir,
    )


if __name__ == "__main__":
    cli()
