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
from scipy.optimize import minimize_scalar
from scipy.special import betaln
from scipy.stats import gamma, nbinom, poisson
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
        "claim_cap": claim_cap,
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
    if values.ndim != 1 or values.size == 0 or values.shape != weights.shape:
        raise ValueError("values and weights must be aligned nonempty 1D arrays")
    if np.any(~np.isfinite(values)):
        raise ValueError("values must be finite")
    if np.any(~np.isfinite(weights) | (weights <= 0)) or not np.isfinite(weights.sum()):
        raise ValueError("weights must be finite and positive")
    order = np.argsort(values, kind="stable")
    unique, starts = np.unique(values[order], return_index=True)
    cumulative = np.cumsum(np.add.reduceat(weights[order], starts))
    return unique, cumulative / cumulative[-1]


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


def fit_gamma_dispersion(
    y: np.ndarray, mean: np.ndarray, claim_count: np.ndarray
) -> float:
    """Fit base phi with fixed means and Gamma(n/phi, phi*mean/n) likelihood.

    Each policy contributes once: n already scales its dispersion.
    """
    y, mean, claim_count = (np.asarray(v, dtype=float) for v in (y, mean, claim_count))
    if (
        y.ndim != 1
        or y.size == 0
        or y.shape != mean.shape
        or y.shape != claim_count.shape
    ):
        raise ValueError("y, mean and claim_count must be aligned nonempty 1D arrays")
    if any(np.any(~np.isfinite(v) | (v <= 0)) for v in (y, mean, claim_count)):
        raise ValueError("y, mean and claim_count must be finite and positive")
    if np.any(claim_count != np.floor(claim_count)):
        raise ValueError("claim_count must contain integers")
    start = float(np.mean(claim_count * (y / mean - 1) ** 2))
    if not np.isfinite(start) or start <= 0:
        raise ValueError("dispersion fitting needs a positive finite start")

    def objective(log_phi: float) -> float:
        with np.errstate(
            over="ignore", under="ignore", divide="ignore", invalid="ignore"
        ):
            phi = np.exp(log_phi)
            likelihood = gamma.logpdf(
                y, a=claim_count / phi, scale=phi * mean / claim_count
            ).sum()
        return -float(likelihood) if np.isfinite(likelihood) else float("inf")

    result = minimize_scalar(
        objective, bracket=(np.log(start) - np.log(2), np.log(start))
    )
    with np.errstate(over="ignore", under="ignore"):
        dispersion = float(np.exp(result.x))
    if (
        not result.success
        or not np.isfinite(result.fun)
        or not np.isfinite(dispersion)
        or dispersion <= 0
    ):
        raise RuntimeError(f"Gamma dispersion optimization failed: {result.message}")
    return dispersion


def fit_negative_binomial_shape(counts: np.ndarray, mean: np.ndarray) -> float:
    """Fit positive real r at fixed policy-count means; infinity is the Poisson limit.

    Compare the optimized NB likelihood with the exact Poisson likelihood.
    A difference below 1e-8 per policy is numerically indistinguishable.
    """
    counts, mean = (np.asarray(v, dtype=float) for v in (counts, mean))
    if counts.ndim != 1 or counts.size == 0 or counts.shape != mean.shape:
        raise ValueError("counts and mean must be aligned nonempty 1D arrays")
    if np.any(~np.isfinite(counts) | (counts < 0) | (counts != np.floor(counts))):
        raise ValueError("counts must be finite nonnegative integers")
    if np.any(~np.isfinite(mean) | (mean <= 0)):
        raise ValueError("mean must be finite and positive")

    def objective(log_shape: float) -> float:
        shape = np.exp(log_shape)
        # Log beta avoids cancellation between large log-gamma values near Poisson.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            likelihood = (
                -betaln(shape, counts + 1)
                - np.log(shape + counts)
                + counts * (np.log(mean) - log_shape)
                - (shape + counts) * np.log1p(mean / shape)
            ).sum()
        return -float(likelihood) if np.isfinite(likelihood) else float("inf")

    # ponytail: bound r to [1e-8, 1e8]; use a stable asymptotic likelihood if a boundary matters.
    bounds = (np.log(1e-8), np.log(1e8))
    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x):
        raise RuntimeError(f"NB shape optimization failed: {result.message}")
    poisson_nll = -float(poisson.logpmf(counts, mean).sum())
    if not np.isfinite(poisson_nll):
        raise RuntimeError("NB Poisson-limit likelihood is non-finite")
    if poisson_nll <= result.fun + 1e-8 * len(counts):
        return float("inf")
    if not bounds[0] + 1e-4 < result.x < bounds[1] - 1e-4:
        raise RuntimeError("NB shape optimization reached a numerical bound")
    return float(np.exp(result.x))


def _diagnostic_inputs(
    data: pd.DataFrame, mean: pd.Series, component: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Require eligible rows and indexed predictions in exactly the same order."""
    if (
        not isinstance(mean, pd.Series)
        or not data.index.is_unique
        or not mean.index.equals(data.index)
    ):
        raise ValueError(
            "mean must be a Series aligned with the unique supplied row index"
        )
    rows, target, weight = target_and_weight(data, component)
    if len(rows) != len(data) or len(rows) == 0:
        raise ValueError("supply nonempty eligible rows for this component")
    prediction = mean.to_numpy(dtype=float)
    if np.any(~np.isfinite(prediction) | (prediction <= 0)):
        raise ValueError("mean must be finite and positive")
    if np.any(~np.isfinite(target) | (target < 0)) or (
        component == "severity" and np.any(target == 0)
    ):
        raise ValueError("target is outside the component support")
    if np.any(~np.isfinite(weight) | (weight <= 0)) or not np.isfinite(weight.sum()):
        raise ValueError("risk weights must be finite and positive")
    if component in ("frequency", "severity"):
        counts = data["ClaimNb"].to_numpy(dtype=float)
        if np.any(~np.isfinite(counts) | (counts < 0) | (counts != np.floor(counts))):
            raise ValueError("ClaimNb must contain finite nonnegative integers")
    return target, prediction, weight


def _conditional_probability(
    thresholds: np.ndarray,
    mean: np.ndarray,
    weight: np.ndarray,
    component: str,
    dispersion: float,
    tweedie_power: float,
    *,
    survival: bool,
    nb_shape: float | None = None,
) -> np.ndarray:
    """Average conditional probabilities in batches of 8 thresholds x 512 rows."""
    result = np.zeros(len(thresholds))
    for start in range(0, len(mean), 512):
        mu, w = mean[start : start + 512], weight[start : start + 512]
        for offset in range(0, len(thresholds), 8):
            x = thresholds[offset : offset + 8, None]
            if component == "frequency":
                if nb_shape is None or np.isposinf(nb_shape):
                    probability = (poisson.sf if survival else poisson.cdf)(x, w * mu)
                else:
                    probability = (nbinom.sf if survival else nbinom.cdf)(
                        x, nb_shape, nb_shape / (nb_shape + w * mu)
                    )
            elif component == "severity":
                probability = (gamma.sf if survival else gamma.cdf)(
                    x, a=w / dispersion, scale=dispersion * mu / w
                )
            else:
                probability = (tweedie_sf_series if survival else tweedie_cdf_series)(
                    x, w * mu, dispersion * w ** (1 - tweedie_power), tweedie_power
                )
            result[offset : offset + 8] += probability @ (w / weight.sum())
    return result


def _conditional_diagnostic(
    data, mean, component, dispersion, tweedie_power, axes, *, survival, nb_shape=None
):
    """Plot the supplied rows against their fitted mixture, with no fitting or sampling."""
    target, prediction, weight = _diagnostic_inputs(data, mean, component)
    if not np.isfinite(dispersion) or dispersion <= 0:
        raise ValueError("dispersion must be finite and positive")
    if component == "pure_premium" and not 1 < tweedie_power < 2:
        raise ValueError("tweedie_power must lie in (1, 2)")
    if nb_shape is not None and (np.isnan(nb_shape) or nb_shape <= 0):
        raise ValueError("nb_shape must be positive, or infinity for the Poisson limit")
    values = (
        data["ClaimNb"].to_numpy(dtype=float)
        if component == "frequency"
        else data["ClaimAmountCapped"].to_numpy(dtype=float)
        if component == "pure_premium"
        else target
    )
    empirical_x, empirical = (
        weighted_empirical_ccdf if survival else weighted_empirical_cdf
    )(values, weight)
    log_x = component == "severity" or (
        survival and component == "pure_premium" and np.any(values > 0)
    )
    if component == "frequency":
        grid = np.floor(np.linspace(0, values.max() + 1, 128))
    elif survival and np.any(values > 0):
        positive = values > 0
        lower, upper = np.quantile(
            values[positive],
            [0.01, 0.995],
            weights=weight[positive],
            method="inverted_cdf",
        )
        grid = np.geomspace(lower, max(upper, lower * 1.01), 128)
    elif component == "severity":
        grid = np.geomspace(values.min() / 2, values.max() * 1.5, 128)
    else:
        grid = np.linspace(0, max(values.max(), 1.0), 128)
    reference = _conditional_probability(
        grid,
        prediction,
        weight,
        component,
        dispersion,
        tweedie_power,
        survival=survival,
    )
    label = {
        "frequency": "Poisson mixture",
        "severity": f"Gamma mixture (φ={dispersion:.3g})",
        "pure_premium": f"Tweedie mixture (p={tweedie_power:.2g}, φ={dispersion:.3g})",
    }[component]
    observable = {
        "frequency": "ClaimNb",
        "severity": "Capped policy-average severity (EUR)",
        "pure_premium": "Capped policy total (EUR)",
    }[component]
    if survival:
        fig, output_axes = _single_panel_axes(axes, (6.5, 4.5))
        panels = (output_axes,)
        within = (empirical_x >= grid[0]) & (empirical_x <= grid[-1])
        empirical_x, empirical = empirical_x[within], empirical[within]
    else:
        fig, output_axes = _cdf_diagnostic_axes(axes, (11.0, 4.5))
        panels = output_axes
    for index, axis in enumerate(panels):
        axis.step(
            empirical_x,
            empirical,
            where="post",
            color=_EMPIRICAL_COLOR,
            label="Empirical",
        )
        axis.plot(
            grid,
            reference,
            color=_FIT_COLOR,
            label=label,
            drawstyle="steps-post" if component == "frequency" else "default",
        )
        if nb_shape is not None:
            nb = _conditional_probability(
                grid,
                prediction,
                weight,
                component,
                dispersion,
                tweedie_power,
                survival=survival,
                nb_shape=nb_shape,
            )
            nb_label = (
                "NB: Poisson limit (r=∞)"
                if np.isposinf(nb_shape)
                else f"NB mixture (r={nb_shape:.3g})"
            )
            axis.step(grid, nb, where="post", color=_FIT_COLOR_2, label=nb_label)
        if log_x and (survival or index == 1):
            axis.set_xscale("log")
        if survival or index == 1:
            axis.set_yscale("log")
            positive_probability = empirical[empirical > 0]
            if positive_probability.size:
                _set_log_y_observed_floor(axis, float(positive_probability.min()))
        if component == "frequency":
            _set_integer_xaxis(axis, int(values.max()))
        elif survival:
            axis.set_xlim(grid[0], grid[-1])
        axis.set_xlabel(observable)
        axis.set_ylabel("SF: P(Y > x)" if survival else "CDF: P(Y ≤ x)")
        axis.set_title(
            f"{component.replace('_', ' ').title()} (n={len(data):,} policies)"
        )
        axis.legend()
    fig.tight_layout()
    return fig, output_axes


def poisson_cdf_diagnostic(data: pd.DataFrame, mean: pd.Series, axes=None):
    """Exposure-weighted policy counts versus Poisson(Exposure * predicted rate)."""
    return _conditional_diagnostic(
        data, mean, "frequency", 1.0, 1.0, axes, survival=False
    )


def poisson_ccdf_diagnostic(
    data: pd.DataFrame, mean: pd.Series, nb_shape: float, axes=None
):
    """Count survival with a fitted real NB shape; infinity labels the Poisson limit."""
    if nb_shape is None:
        raise ValueError("nb_shape is required")
    return _conditional_diagnostic(
        data, mean, "frequency", 1.0, 1.0, axes, survival=True, nb_shape=nb_shape
    )


def gamma_cdf_diagnostic(
    data: pd.DataFrame, mean: pd.Series, dispersion: float, axes=None
):
    """Claim-weighted capped policy-average severity versus Gamma(n/phi, phi*mean/n)."""
    return _conditional_diagnostic(
        data, mean, "severity", dispersion, 2.0, axes, survival=False
    )


def gamma_ccdf_diagnostic(
    data: pd.DataFrame, mean: pd.Series, dispersion: float, axes=None
):
    """Survival for the same capped policy averages and fitted Gamma reference as the CDF."""
    return _conditional_diagnostic(
        data, mean, "severity", dispersion, 2.0, axes, survival=True
    )


def tweedie_cdf_diagnostic(
    data: pd.DataFrame,
    mean: pd.Series,
    tweedie_power: float,
    dispersion: float,
    axes=None,
):
    """Exposure-weighted capped totals; mean=e*rate, dispersion=phi*e**(1-p)."""
    return _conditional_diagnostic(
        data, mean, "pure_premium", dispersion, tweedie_power, axes, survival=False
    )


def tweedie_ccdf_diagnostic(
    data: pd.DataFrame,
    mean: pd.Series,
    tweedie_power: float,
    dispersion: float,
    axes=None,
):
    """Survival for the same capped policy totals and fitted Tweedie reference as the CDF."""
    return _conditional_diagnostic(
        data, mean, "pure_premium", dispersion, tweedie_power, axes, survival=True
    )


def _tweedie_probability_series(
    x: np.ndarray,
    mu: float | np.ndarray,
    phi: float | np.ndarray,
    tweedie_power: float,
    max_terms: int,
    tol: float,
    *,
    survival: bool,
) -> np.ndarray:
    """Sum conditional Gamma probabilities with an omitted-Poisson-mass bound."""
    if not (1 < tweedie_power < 2):
        raise ValueError("tweedie_power must lie in (1, 2)")
    if not isinstance(max_terms, (int, np.integer)) or max_terms < 1:
        raise ValueError("max_terms must be a positive integer")
    if not np.isfinite(tol) or not (0 < tol < 1):
        raise ValueError("tol must lie in (0, 1)")
    x_arr, mu_arr, phi_arr = np.broadcast_arrays(
        np.atleast_1d(np.asarray(x, dtype=float)),
        np.asarray(mu, dtype=float),
        np.asarray(phi, dtype=float),
    )
    if np.any(np.isnan(x_arr)):
        raise ValueError("x must not contain NaN")
    if np.any(~np.isfinite(mu_arr) | (mu_arr <= 0)) or np.any(
        ~np.isfinite(phi_arr) | (phi_arr <= 0)
    ):
        raise ValueError("mu and phi must be finite and positive")
    p = tweedie_power
    alpha = (2 - p) / (p - 1)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        beta = phi_arr * (p - 1) * mu_arr ** (p - 1)
        lam = mu_arr ** (2 - p) / (phi_arr * (2 - p))
    if np.any(~np.isfinite(beta) | (beta <= 0)) or np.any(
        ~np.isfinite(lam) | (lam <= 0)
    ):
        raise ValueError(
            "compound Poisson-Gamma parameters must be finite and positive"
        )
    result = np.zeros(x_arr.shape)
    result[x_arr < 0] = float(survival)
    result[np.isposinf(x_arr)] = float(not survival)
    zero = x_arr == 0
    result[zero] = -np.expm1(-lam[zero]) if survival else np.exp(-lam[zero])
    active = np.isfinite(x_arr) & (x_arr > 0)
    if not survival:
        result[active] = np.exp(-lam[active])
    gamma_probability = gamma.sf if survival else gamma.cdf
    for k in range(1, max_terms + 1):
        if not np.any(active):
            return result
        intensity = lam[active]
        threshold = x_arr[active]
        scale = beta[active]
        result[active] += np.exp(poisson.logpmf(k, intensity)) * gamma_probability(
            threshold, a=k * alpha, scale=scale
        )
        log_remainder = poisson.logsf(k, intensity)
        if not survival:
            # Gamma CDFs decrease with k, tightening the omitted-mass bound.
            log_remainder += gamma.logcdf(threshold, a=(k + 1) * alpha, scale=scale)
        with np.errstate(divide="ignore"):
            log_target = np.log(tol) + np.log(result[active])
        # A zero sum can stop only when the bound is below float representation.
        log_target = np.where(
            result[active] > 0, log_target, np.log(np.nextafter(0.0, 1.0))
        )
        active[active] = log_remainder > log_target
    if np.any(active):
        raise RuntimeError(
            f"Tweedie probability series did not converge in {max_terms} terms"
        )
    return result


def tweedie_cdf_series(
    x: np.ndarray,
    mu: float | np.ndarray,
    phi: float | np.ndarray,
    tweedie_power: float,
    max_terms: int = 10_000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Return P(Y <= x), including the compound Poisson mass at zero.

    Inputs broadcast together. Sum Poisson log-PMF weights times Gamma CDFs
    until the omitted probability is at most ``tol`` times the partial sum.
    Raise if ``max_terms`` is exhausted before convergence.
    """
    return _tweedie_probability_series(
        x, mu, phi, tweedie_power, max_terms, tol, survival=False
    )


def tweedie_sf_series(
    x: np.ndarray,
    mu: float | np.ndarray,
    phi: float | np.ndarray,
    tweedie_power: float,
    max_terms: int = 10_000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Return P(Y > x) by summing Gamma survival probabilities directly.

    Inputs broadcast together. The omitted Poisson mass bounds the error
    relative to the partial sum, including tiny tails. No CDF subtraction or
    probability floor is used; unachieved convergence raises RuntimeError.
    """
    return _tweedie_probability_series(
        x, mu, phi, tweedie_power, max_terms, tol, survival=True
    )


def conditional_dispersion_table(
    data: pd.DataFrame, mean: pd.Series, by: str, response: str, tweedie_power: float
) -> pd.DataFrame:
    """Report fixed-power residual dispersion, dividing by row count, not risk volume."""
    target, prediction, weight = _diagnostic_inputs(data, mean, response)
    power = {"frequency": 1.0, "severity": 2.0, "pure_premium": tweedie_power}[response]
    if response == "pure_premium" and not 1 < power < 2:
        raise ValueError("tweedie_power must lie in (1, 2)")
    frame = pd.DataFrame(
        {
            by: data[by],
            "risk_volume": weight,
            "observed_rate": weight * target,
            "predicted_rate": weight * prediction,
            "dispersion": weight * (target - prediction) ** 2 / prediction**power,
        }
    )
    grouped = frame.groupby(by, observed=True, sort=True, dropna=False)
    table = grouped.sum()
    table.insert(0, "n", grouped.size())
    table[["observed_rate", "predicted_rate"]] = table[
        ["observed_rate", "predicted_rate"]
    ].div(table["risk_volume"], axis=0)
    table["dispersion"] /= table["n"]
    return table.reset_index()


def conditional_dispersion_diagnostic(
    data: pd.DataFrame,
    predictions: dict[str, pd.Series],
    by: str,
    tweedie_power: float,
    dispersions: dict[str, float],
    axes=None,
):
    """Compare grouped training residual dispersion with fixed family references."""
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    else:
        fig = axes[0].figure
    for axis, component in zip(axes, TARGET_SPECS, strict=True):
        rows, _, _ = target_and_weight(data, component)
        table = conditional_dispersion_table(
            rows, predictions[component], by, component, tweedie_power
        )
        reference = 1.0 if component == "frequency" else dispersions[component]
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError("dispersion must be finite and positive")
        x = np.arange(len(table))
        axis.scatter(
            x, table["dispersion"], color=_EMPIRICAL_COLOR, label="Residual dispersion"
        )
        axis.axhline(
            reference, color=_FIT_COLOR, ls="--", label=f"Reference: {reference:.3g}"
        )
        axis.set_xticks(x, table[by].astype(str))
        axis.set_xlabel(by)
        axis.set_ylabel("Mean scaled residual square")
        axis.set_title(f"{component.replace('_', ' ').title()} (n={len(rows):,})")
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


def tweedie_log_likelihood(
    y: np.ndarray,
    mean: np.ndarray,
    exposure: np.ndarray,
    tweedie_power: float,
    dispersion: float,
) -> float:
    """Sum log f_p(y_i; mean_i, dispersion / exposure_i) for rate targets.

    Scale each rate and mean by a_i = exposure_i ** (1 / (2 - p)) so GLUM
    can evaluate all rows with one scalar dispersion and unit likelihood
    weights. Only positive observations receive the log(a_i) Jacobian;
    zero observations retain their discrete probability mass.
    """
    if not (1 < tweedie_power < 2):
        raise ValueError("tweedie_power must lie in (1, 2)")
    y, mean, exposure = (np.asarray(v, dtype=float) for v in (y, mean, exposure))
    if y.ndim != 1 or y.size == 0 or y.shape != mean.shape or y.shape != exposure.shape:
        raise ValueError("y, mean and exposure must be aligned nonempty 1D arrays")
    if np.any(~np.isfinite(y) | (y < 0)):
        raise ValueError("y must be finite and nonnegative")
    if (
        np.any(~np.isfinite(mean) | (mean <= 0))
        or np.any(~np.isfinite(exposure) | (exposure <= 0))
        or not np.isfinite(dispersion)
        or dispersion <= 0
    ):
        raise ValueError("mean, exposure and dispersion must be finite and positive")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        log_scale = np.log(exposure) / (2 - tweedie_power)
        scale = np.exp(log_scale)
        scaled_y, scaled_mean = scale * y, scale * mean
    if (
        np.any(~np.isfinite(scale) | (scale <= 0))
        or np.any(~np.isfinite(scaled_mean) | (scaled_mean <= 0))
        or np.any(~np.isfinite(scaled_y) | ((y > 0) & (scaled_y <= 0)))
    ):
        raise ValueError("exposure scaling produced invalid transformed values")
    likelihood = (
        TweedieDistribution(tweedie_power).log_likelihood(
            scaled_y,
            scaled_mean,
            sample_weight=np.ones_like(y),
            dispersion=float(dispersion),
        )
        + log_scale[y > 0].sum()
    )
    if not np.isfinite(likelihood):
        raise RuntimeError("Tweedie likelihood produced a non-finite value")
    return float(likelihood)


def fit_tweedie_dispersion(
    y: np.ndarray,
    mean: np.ndarray,
    exposure: np.ndarray,
    tweedie_power: float,
) -> float:
    """Fit base dispersion by rate likelihood, keeping predicted means fixed.

    The mean exposure-adjusted Pearson residual square is only a starting
    estimate. Optimize log dispersion and raise on failure; no moment fallback.
    """
    y, mean, exposure = (np.asarray(v, dtype=float) for v in (y, mean, exposure))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        start = float(np.mean(exposure * (y - mean) ** 2 / mean**tweedie_power))
    if not np.isfinite(start) or start <= 0 or not np.any(y > 0):
        raise ValueError(
            "dispersion fitting needs positive loss and a positive finite start"
        )

    def objective(log_dispersion: float) -> float:
        with np.errstate(over="ignore", under="ignore"):
            dispersion = float(np.exp(log_dispersion))
        if not np.isfinite(dispersion) or dispersion <= 0:
            return float("inf")
        return -tweedie_log_likelihood(y, mean, exposure, tweedie_power, dispersion)

    log_start = np.log(start)
    result = minimize_scalar(objective, bracket=(log_start - np.log(2), log_start))
    with np.errstate(over="ignore", under="ignore"):
        dispersion = float(np.exp(result.x))
    if (
        not result.success
        or not np.isfinite(result.fun)
        or not np.isfinite(dispersion)
        or dispersion <= 0
    ):
        raise RuntimeError(f"Tweedie dispersion optimization failed: {result.message}")
    return dispersion


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
        dispersion = fit_tweedie_dispersion(y_train, train_prediction, w_train, power)
        log_likelihood = tweedie_log_likelihood(
            y_validation, validation_prediction, w_validation, power, dispersion
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
    """Aggregate tied scores first, then split blocks across equal-exposure bins."""
    y_true_rate = np.asarray(y_true_rate, dtype=float)
    y_pred_rate = np.asarray(y_pred_rate, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if (
        exposure.ndim != 1
        or exposure.size == 0
        or any(values.shape != exposure.shape for values in (y_true_rate, y_pred_rate))
    ):
        raise ValueError("rates and exposure must be aligned nonempty 1D arrays")
    if not np.all(np.isfinite(exposure) & (exposure > 0)) or not np.isfinite(
        exposure.sum()
    ):
        raise ValueError("exposure must be finite and positive")
    if any(
        np.any(~np.isfinite(values) | (values < 0))
        for values in (y_true_rate, y_pred_rate)
    ):
        raise ValueError("rates must be finite and nonnegative")

    order = np.argsort(y_pred_rate, kind="stable")
    _, starts = np.unique(y_pred_rate[order], return_index=True)
    cumulative_exposure = np.r_[
        0.0, np.cumsum(np.add.reduceat(exposure[order], starts))
    ]
    cumulative_observed = np.r_[
        0.0, np.cumsum(np.add.reduceat(exposure[order] * y_true_rate[order], starts))
    ]
    cumulative_predicted = np.r_[
        0.0, np.cumsum(np.add.reduceat(exposure[order] * y_pred_rate[order], starts))
    ]
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
    """Bin low-to-high A/B ratios, distributing tied blocks proportionally."""
    y_true_rate = np.asarray(y_true_rate, dtype=float)
    model_a_prediction = np.asarray(model_a_prediction, dtype=float)
    model_b_prediction = np.asarray(model_b_prediction, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if (
        exposure.ndim != 1
        or exposure.size == 0
        or any(
            values.shape != exposure.shape
            for values in (y_true_rate, model_a_prediction, model_b_prediction)
        )
    ):
        raise ValueError("rates and exposure must be aligned nonempty 1D arrays")
    if np.any(~np.isfinite(y_true_rate) | (y_true_rate < 0)):
        raise ValueError("observed rates must be finite and nonnegative")
    if not np.all(np.isfinite(exposure) & (exposure > 0)) or not np.isfinite(
        exposure.sum()
    ):
        raise ValueError("exposure must be finite and positive")
    if not np.all(np.isfinite(model_a_prediction) & (model_a_prediction > 0)):
        raise ValueError("model_a_prediction must be finite and positive")
    if not np.all(np.isfinite(model_b_prediction) & (model_b_prediction > 0)):
        raise ValueError("model_b_prediction must be finite and positive")
    with np.errstate(over="ignore", under="ignore"):
        ratio = model_a_prediction / model_b_prediction
    if np.any(~np.isfinite(ratio) | (ratio <= 0)):
        raise ValueError("prediction ratios must be finite and positive")
    order = np.argsort(ratio, kind="stable")
    _, starts = np.unique(ratio[order], return_index=True)
    cumulative_exposure = np.r_[
        0.0, np.cumsum(np.add.reduceat(exposure[order], starts))
    ]
    boundaries = np.linspace(0, cumulative_exposure[-1], n_bins + 1)
    bin_exposure = np.diff(boundaries)

    def rates(values: np.ndarray) -> np.ndarray:
        cumulative = np.r_[
            0.0, np.cumsum(np.add.reduceat(exposure[order] * values[order], starts))
        ]
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
    frame = pd.DataFrame({"band": bands, "weight": weight, "observed": target * weight})
    for name, prediction in predictions.items():
        frame[name] = np.asarray(prediction) * weight
    grouped = frame.groupby("band", sort=True, observed=True).sum()
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
            axis.set_xticks(x, calibration.index.astype(str), rotation=45, ha="right")
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
    - ``"diag"`` uses the predicted span on x and includes zero on observed y.
    - ``"residual"`` uses ``(xmin, xmax)`` on x and the symmetric predicted span
      on y (residual panel).
    - ``"auto"`` leaves hexbinning and limits alone (log panels).
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    artists = []
    for axis, (name, (x, y)) in zip(axes.flat, xy_by_model.items()):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 1 or x.shape != y.shape:
            raise ValueError("x and y must be aligned 1D arrays")
        visible = np.isfinite(x) & np.isfinite(y)
        if logx:
            visible &= x > 0
        if logy:
            visible &= y > 0
        x, y = x[visible], y[visible]
        axis.set_title(name)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_xscale("log" if logx else "linear")
        axis.set_yscale("log" if logy else "linear")
        if x.size == 0:
            axis.text(
                0.5, 0.5, "No eligible policies", transform=axis.transAxes, ha="center"
            )
            continue
        xlo, xhi = float(x.min()), float(x.max())
        if xlo == xhi:
            padding = max(abs(xlo) * 0.05, 0.5)
            xlo, xhi = xlo - padding, xhi + padding
        if extent_mode == "diag" and not logx and not logy:
            ylo, yhi = min(0.0, xlo), max(0.0, xhi)
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
    for axis in list(axes.flat)[len(xy_by_model) :]:
        axis.set_visible(False)
    vmax = max(
        (
            float(artist.get_array().max())
            for artist in artists
            if artist.get_array().size
        ),
        default=1.0,
    )
    norm = mcolors.LogNorm(1.0, max(2.0, vmax))
    for artist in artists:
        artist.set_norm(norm)
    if artists:
        colorbar = fig.colorbar(artists[0], ax=axes, fraction=0.03)
        colorbar.set_label("policies per bin")
    plt.show()
    return fig, axes


def print_frame(title: str, frame: pd.DataFrame) -> None:
    """Render every tutorial table through the shared Rich console."""
    shown = (
        frame.reset_index()
        if frame.index.name is not None
        or not frame.index.equals(pd.RangeIndex(len(frame)))
        else frame
    )
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
