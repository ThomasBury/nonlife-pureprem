# %% [markdown]
# # Exposure Handling for Tweedie & Poisson Models: A Practical Tutorial
#
# **Audience:** Pricing actuaries, reserving analysts, and data scientists fitting
# frequency and pure premium models on insurance data.
#
# **What this script demonstrates:**
#
# In actuarial pricing, exposure usually represents risk volume: policy-years,
# vehicle-years, contract duration, distance driven, or another measure of time at risk.
# Under this convention, exposure scales the claim-generating process.
#
# For frequency:
#
# - Counts: `E[ClaimNb_i] = Exposure_i * lambda_i`
# - Rates: `Frequency_i = ClaimNb_i / Exposure_i`
# - Rate model weight: `sample_weight = Exposure_i`
#
# For pure premium:
#
# - Totals: `E[ClaimAmount_i] = Exposure_i * mu_i`
# - Rates: `PurePremium_i = ClaimAmount_i / Exposure_i`
# - Rate model weight: `sample_weight = Exposure_i`
#
# This tutorial uses the actuarial compound-Poisson exposure convention:
#
#     target = rate
#     sample_weight = Exposure
#
# Mathematical details and alternative variance conventions are discussed in the
# accompanying PDF notes. This notebook focuses on implementation, diagnostics,
# and model validation on MTPL data.


# %% [markdown]
# ## Distribution Misspecification: What It Means and Why You Should Care
#
# Distribution misspecification happens when the loss function you minimize assumes
# a variance structure that is not aligned with the data-generating process.
#
# In a quasi-likelihood GLM, we do not need to specify the full conditional
# distribution. We specify:
#
# - the conditional mean,
# - the variance function,
# - the observation weights.
#
# For a Tweedie pure premium model, the rate target is:
#
#     PurePremium_i = ClaimAmount_i / Exposure_i
#
# Under the actuarial compound-Poisson exposure convention:
#
#     Var(PurePremium_i | x_i) = phi * mu_i^p / Exposure_i
#
# Therefore the natural quasi-likelihood weight for the rate model is:
#
#     sample_weight = Exposure_i
#
# **Why p matters for insurance data:**
#
# MTPL pure premium data typically has many exact zeros and a heavy right tail on
# positive losses. This is the empirical pattern targeted by Tweedie models with
# `1 < p < 2`.
#
# - `p = 1` corresponds to the Poisson boundary.
# - `p = 2` corresponds to the Gamma boundary.
# - `1 < p < 2` represents the compound-Poisson-Gamma region.
#
# The value of `p` controls how the loss balances zeros and large positive claims.
# It should be estimated or validated, not chosen blindly.
#
# **Important:**
#
# The sample weight is not a business preference. It encodes the assumed variance
# scaling. In this tutorial, because exposure is treated as actuarial risk volume,
# pure premium and frequency rate models use:
#
#     sample_weight = Exposure
#
# **Why regularization can mask misspecification:**
#
# A well-tuned L2 penalty can shrink coefficients and reduce the visible impact of
# misspecification on hold-out deviance. The model may look acceptable on test data
# while still being structurally miscalibrated. Use calibration checks, lift charts,
# Lorenz curves, and portfolio-level totals in addition to deviance.

# %% [markdown]
# ## Critical Mistakes to Avoid
#
# 1. **Using unweighted rates.**
#
#    A policy with 1 year of exposure is more informative about the annualized rate
#    than a policy with 2 days of exposure. For rate models, use:
#
#        sample_weight = Exposure
#
# 2. **Treating exposure as an ordinary feature.**
#
#    Exposure is not a rating factor in the usual sense. It is the amount of risk
#    observed. For rate models, divide the total by exposure and use exposure as
#    the observation weight. For total-count models, use a log-exposure offset
#    when the implementation supports it.
#
# 3. **Leaving `boost_from_average = True` when comparing LightGBM offset and rate
#    formulations.**
#
#    LightGBM computes a global starting value from the labels. Totals and rates
#    have different scales, so they can start from different base scores. When
#    comparing formulations, control the initialization explicitly.
#
# 4. **Not estimating or validating the Tweedie power p.**
#
#    Hardcoding `p = 1.5` just because the target is Tweedie is poor practice.
#    Estimate p or run a sensitivity analysis. MTPL data with many zeros often
#    leads to p closer to the Poisson boundary, but the positive tail pushes p
#    above 1.
#
# 5. **Evaluating with unweighted metrics on rates.**
#
#    Pure premium and frequency are exposure-normalized targets. Evaluation should
#    also be exposure-aware. Use exposure-weighted Tweedie deviance, Poisson
#    deviance, calibration plots, and aggregate totals.
#
# 6. **Forgetting to floor tiny exposures.**
#
#    Dividing by a very small exposure creates extreme rate values. Always apply
#    a small exposure floor before constructing rate targets.

# %%
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families.family import Tweedie as TweedieFamily
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import PoissonRegressor, TweedieRegressor
from sklearn.metrics import (
    auc,
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    mean_tweedie_deviance,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)

# Optional LightGBM
try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    warnings.warn("lightgbm not found. LightGBM parts will be skipped.", RuntimeWarning)

# Optional XGBoost
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("xgboost not found. XGBoost parts will be skipped.", RuntimeWarning)

# Optional Rich terminal rendering
try:
    from rich import box
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None
    warnings.warn("rich not found. Falling back to plain text tables.", RuntimeWarning)


# Config
# ~93% zeros in MTPL --> p close to 1 (Poisson); Gamma tail on positives pushes p above 1.
# P_INIT is the starting guess for the Pearson estimator; P will be reassigned after estimation.
P_INIT = 1.35
P = P_INIT
# Small ridge penalty for numerical stability with the dense OHE design.
ALPHA = 0.1
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SAMPLES = None  # set to e.g. 200_000 for faster runs; None for full dataset

# Gradient-boosting hyperparameters, tuned for Tweedie / Poisson pure premium on
# heavy-tailed insurance data. Two things dominate: a large floor on the number
# of observations per leaf (so outliers in the right tail can't drive tiny leaves)
# and a light L2 penalty on leaf values. Smaller learning rate + early stopping
# is the standard Tweedie recipe.
LEARNING_RATE = 0.05
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50

# LightGBM-specific
LGB_NUM_LEAVES = 63
LGB_MIN_DATA_IN_LEAF = 100
LGB_FEATURE_FRACTION = 0.8
LGB_BAGGING_FRACTION = 0.8
LGB_BAGGING_FREQ = 5
LGB_LAMBDA_L2 = 0.1

# XGBoost-specific
XGB_MAX_DEPTH = 6
XGB_MIN_CHILD_WEIGHT = 100
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_REG_LAMBDA = 0.1

# CatBoost-specific
CB_DEPTH = 6
CB_L2_LEAF_REG = 3.0
CB_SUBSAMPLE = 0.8

EXPOSURE_FLOOR = 1/366
# Control LightGBM's base score initialization. When False, LightGBM starts from
# the mean of the labels rather than zero, which can differ between the frequency
# offset variant and the rate variant. We disable it for consistent comparisons.
LGB_DISABLE_BOOST_FROM_AVERAGE = True
np.random.seed(RANDOM_STATE)

# %% [markdown]
# ## Exposure helpers
#
# The helpers below keep the implementation aligned with the actuarial exposure
# convention used throughout this tutorial:
#
# - `rate_weights(exposure)` returns `Exposure`;
# - `log_exposure_offset(exposure)` returns the additive log-offset used in count
#   or total models when the implementation supports offsets.
#
# The mathematical discussion of alternative Tweedie variance conventions is kept
# in the PDF notes. This notebook uses the standard actuarial rate formulation:
#
#     target = total / exposure
#     sample_weight = exposure


# %%
def rate_weights(exposure: np.ndarray) -> np.ndarray:
    """Return actuarial rate weights: Exposure."""
    exposure = np.asarray(exposure, dtype=float)
    return exposure


def log_exposure_offset(
    exposure: np.ndarray, floor: float = EXPOSURE_FLOOR
) -> np.ndarray:
    """Return log(exposure) with a safety floor for tiny exposures."""
    exposure = np.asarray(exposure, dtype=float)
    return np.log(np.fmax(exposure, floor))


# %% [markdown]
# ## Utility functions (output formatting, plot helpers)
#
# *You can skip this section.* These are Rich table formatters, Lorenz/lift-chart
# functions, and data loading helpers. The tutorial content starts at
# "Data loading and preprocessing" below.


# %%
def emit(message: str, style: str | None = None) -> None:
    """Print a message with Rich when available, else fall back to plain print."""
    if console is None:
        print(message)
        return
    console.print(message, style=style)


def _format_scalar(value: object) -> str:
    """Format scalar values for readable terminal tables."""
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return "-"
    except TypeError:
        pass

    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"

    if isinstance(value, (np.floating, float)):
        value = float(value)
        abs_value = abs(value)
        if abs_value >= 10_000:
            return f"{value:,.2f}"
        if abs_value >= 1:
            return f"{value:,.4f}"
        if abs_value == 0:
            return "0.0000"
        return f"{value:.6f}"

    return str(value)


def _row_style_from_model(model_name: str) -> str:
    """Use model naming conventions to assign a meaningful Rich row style."""
    lower_name = model_name.lower()
    if "observed" in lower_name:
        return "bold white"
    if "offset" in lower_name:
        return "bold cyan"
    if "lgb" in lower_name:
        return "yellow"
    return ""


def print_summary_table(title: str, rows: tuple[tuple[str, object], ...]) -> None:
    """Print a compact two-column summary table."""
    if console is None:
        print(f"\n=== {title} ===")
        for key, value in rows:
            print(f"{key}: {_format_scalar(value)}")
        return

    table = Table(
        title=title, box=box.ROUNDED, header_style="bold magenta", show_edge=True
    )
    table.add_column("Item", style="bold")
    table.add_column("Value", justify="right", style="cyan")
    for key, value in rows:
        table.add_row(key, _format_scalar(value))
    console.print(table)


def print_dataframe_table(
    df: pd.DataFrame,
    title: str,
    model_col: str = "model",
    caption: str | None = None,
) -> None:
    """Render a DataFrame as a Rich table with sensible numeric formatting."""
    if console is None:
        print(f"\n=== {title} ===")
        if caption:
            print(caption)
        with pd.option_context("display.max_rows", 100, "display.width", 180):
            print(df)
        return

    table = Table(
        title=title,
        caption=caption,
        box=box.SIMPLE_HEAVY,
        header_style="bold magenta",
        row_styles=["none", "dim"],
        show_edge=True,
    )

    numeric_cols = {col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])}
    for col in df.columns:
        justify = "right" if col in numeric_cols else "left"
        style = "bold" if col == model_col else ""
        table.add_column(
            str(col), justify=justify, style=style, no_wrap=(col == model_col)
        )

    for _, row in df.iterrows():
        row_style = (
            _row_style_from_model(str(row[model_col]))
            if model_col in df.columns
            else ""
        )
        table.add_row(
            *[_format_scalar(row[col]) for col in df.columns], style=row_style
        )

    console.print(table)


def print_grouped_metrics_tables(metrics_df: pd.DataFrame, title_prefix: str) -> None:
    """Print one metrics table per evaluation weighting."""
    metrics_reset = metrics_df.reset_index().copy()
    weight_labels = {
        "Exposure": "Business weighting: exposure",
    }

    for eval_weight, subset in metrics_reset.groupby("eval_weight", sort=False):
        display_df = subset.drop(columns=["eval_weight"]).copy()
        display_df = display_df.sort_values("model").reset_index(drop=True)
        caption = str(weight_labels.get(eval_weight, eval_weight))
        print_dataframe_table(
            display_df,
            title=f"{title_prefix} [{eval_weight}]",
            caption=caption,
        )


def build_aggregate_comparison_df(
    observed_label: str,
    observed_value: float,
    rows: tuple[tuple[str, float], ...],
    value_col: str,
) -> pd.DataFrame:
    """Build an aggregate comparison table with absolute and relative error columns."""
    table_rows = [(observed_label, observed_value, np.nan, np.nan)]
    for model_name, predicted_value in rows:
        abs_error = predicted_value - observed_value
        rel_error_pct = (
            100.0 * abs_error / observed_value if observed_value != 0 else np.nan
        )
        table_rows.append((model_name, predicted_value, abs_error, rel_error_pct))

    return pd.DataFrame(
        table_rows,
        columns=["model", value_col, "abs_error", "rel_error_pct"],
    )


# %% [markdown]
# ## Data loading utilities


# %%
def load_mtpl2(n_samples: int | None = None) -> pd.DataFrame:
    """Fetch French MTPL (freq + sev), aggregate severity to totals, clean columns.

    Parameters
    ----------
    n_samples : Optional[int], optional
        Number of samples to load, by default None (all samples).

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame with aggregated totals.
    """
    # freMTPL2freq
    df_freq = fetch_openml(data_id=41214, as_frame=True).data
    df_freq["IDpol"] = df_freq["IDpol"].astype(int)
    df_freq.set_index("IDpol", inplace=True)
    # freMTPL2sev
    df_sev = fetch_openml(data_id=41215, as_frame=True).data
    df_sev = df_sev.groupby("IDpol").sum()

    df = df_freq.join(df_sev, how="left")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)

    # strip quotes in string columns
    for col in df.columns[[t is object for t in df.dtypes.values]]:
        df[col] = df[col].str.strip("'")

    return df.iloc[:n_samples]


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning logic and derive targets.

    Same cleaning logic as sklearn example, plus derived targets.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to clean.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame with derived targets.
    """
    df = df.copy()
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(upper=1)
    df["ClaimAmount"] = df["ClaimAmount"].clip(upper=200_000)
    df.loc[(df["ClaimAmount"] == 0) & (df["ClaimNb"] >= 1), "ClaimNb"] = 0

    # Derived rate targets use a small floor to avoid exploding values when
    # exposure is positive but extremely close to zero.
    exposure_safe = np.fmax(df["Exposure"].to_numpy(dtype=float), EXPOSURE_FLOOR)
    df["PurePremium"] = df["ClaimAmount"] / exposure_safe
    df["Frequency"] = df["ClaimNb"] / exposure_safe
    return df


# %% [markdown]
# ## Feature engineering


# %%
def build_column_transformer() -> ColumnTransformer:
    """Build a column transformer for preprocessing features.

    Creates a ColumnTransformer with binning for numeric features,
    one-hot encoding for categorical features, and log scaling for density.

    Returns
    -------
    ColumnTransformer
        The configured column transformer.
    """
    # Make OHE robust to unseen categories in test
    ohe = OneHotEncoder(handle_unknown="ignore")
    log_scale_transformer = make_pipeline(
        FunctionTransformer(func=np.log), StandardScaler()
    )
    column_trans = ColumnTransformer(
        transformers=[
            (
                "binned_numeric",
                KBinsDiscretizer(n_bins=10, encode="onehot-dense", strategy="quantile"),
                # Use quantile binning to handle skewed distributions like age.
                # Each bin will contain approximately the same number of samples.
                ["VehAge", "DrivAge"],
            ),
            (
                "onehot_categorical",
                ohe,
                ["VehBrand", "VehPower", "VehGas", "Region", "Area"],
            ),
            ("passthrough_numeric", "passthrough", ["BonusMalus"]),
            ("log_scaled_numeric", log_scale_transformer, ["Density"]),
        ],
        remainder="drop",
    )
    return column_trans


# %% [markdown]
# ## Plot helpers (rate-level calibration, Lorenz)


# %%
def _get_aggregated_rates(
    df: pd.DataFrame, feature: str, weight_name: str, rate_values: np.ndarray
) -> pd.DataFrame:
    """Aggregate rates by a feature, exposure-weighted.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    feature : str
        The feature to group by.
    weight_name : str
        The column name for weights.
    rate_values : np.ndarray
        The rate values to aggregate.

    Returns
    -------
    pd.DataFrame
        The aggregated DataFrame with rates.
    """
    w = df[weight_name].to_numpy()
    tmp = pd.DataFrame(
        {
            feature: df[feature].to_numpy(),
            "w": w,
            "rate_total": rate_values * w,
        }
    )
    grp = tmp.groupby(feature)[["w", "rate_total"]].sum()
    grp["rate"] = grp["rate_total"] / grp["w"].replace(0, np.nan)
    return grp


def lorenz_curve(
    y_true_rate: np.ndarray, y_pred_rate: np.ndarray, exposure: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Lorenz curve for exposure-weighted rates.

    Parameters
    ----------
    y_true_rate : np.ndarray
        True rate values.
    y_pred_rate : np.ndarray
        Predicted rate values.
    exposure : np.ndarray
        Exposure values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Cumulative exposure and cumulative true rates.
    """
    y_true_rate = np.asarray(y_true_rate)
    y_pred_rate = np.asarray(y_pred_rate)
    exposure = np.asarray(exposure)

    order = np.argsort(y_pred_rate)
    ex = exposure[order]
    num = np.cumsum(ex * y_true_rate[order])
    num = num / num[-1] if num[-1] > 0 else num
    den = np.cumsum(ex) / np.sum(ex)
    return den, num


def gini_coefficient(
    y_true_rate: np.ndarray, y_pred_rate: np.ndarray, exposure: np.ndarray
) -> float:
    """Gini = 1 - 2 * AUC(Lorenz curve)."""
    cx, cy = lorenz_curve(y_true_rate, y_pred_rate, exposure)
    return 1.0 - 2.0 * auc(cx, cy)


def check_lorenz_dominance(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Pairwise dominance check: A dominates B if A's Lorenz is everywhere >= B's.

    Returns a DataFrame with columns [model_A, model_B, dominates, n_crossings].
    """
    names = list(curves.keys())
    rows = []
    for i, a in enumerate(names):
        cx_a, cy_a = curves[a]
        for b in names[i + 1 :]:
            cx_b, cy_b = curves[b]
            # Interpolate both onto a common grid
            grid = np.union1d(cx_a, cx_b)
            cy_a_i = np.interp(grid, cx_a, cy_a)
            cy_b_i = np.interp(grid, cx_b, cy_b)
            diff = cy_a_i - cy_b_i
            sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
            if np.all(diff >= -1e-12):
                dom = f"{a} dominates"
            elif np.all(diff <= 1e-12):
                dom = f"{b} dominates"
            else:
                dom = "no dominance (crossings)"
            rows.append(
                {
                    "model_A": a,
                    "model_B": b,
                    "dominance": dom,
                    "n_crossings": int(sign_changes),
                }
            )
    return pd.DataFrame(rows)


def _compute_lift_table(
    y_true_rate: np.ndarray,
    y_pred_rate: np.ndarray,
    exposure: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin by predicted risk into deciles; compute exposure-weighted observed & predicted.

    Averages are sum(rate * exposure) / sum(exposure) in each bin, which for
    frequency = sum(ClaimNb) / sum(Exposure) and for pure premium =
    sum(ClaimAmount) / sum(Exposure), i.e. the actuarially correct aggregate.
    """
    y_true_rate = np.asarray(y_true_rate, dtype=float)
    y_pred_rate = np.asarray(y_pred_rate, dtype=float)
    exposure = np.asarray(exposure, dtype=float)

    # Decile labels 1..n_bins based on predicted values
    bin_labels = pd.qcut(y_pred_rate, q=n_bins, labels=False, duplicates="drop") + 1

    tmp = pd.DataFrame(
        {
            "bin": bin_labels,
            "exposure": exposure,
            "obs_total": y_true_rate * exposure,
            "pred_total": y_pred_rate * exposure,
        }
    )
    grp = tmp.groupby("bin").agg(
        exposure=("exposure", "sum"),
        obs_total=("obs_total", "sum"),
        pred_total=("pred_total", "sum"),
        n_policies=("bin", "size"),
    )
    grp["obs_rate"] = grp["obs_total"] / grp["exposure"]
    grp["pred_rate"] = grp["pred_total"] / grp["exposure"]
    return grp


def lift_chart(
    y_true_rate: np.ndarray,
    y_pred_rate: np.ndarray,
    exposure: np.ndarray,
    n_bins: int = 10,
    title: str = "Lift chart",
    filename: str | None = None,
) -> pd.DataFrame:
    """Plot and return the single-model lift chart.

    Returns the lift table for further inspection.
    """
    tbl = _compute_lift_table(y_true_rate, y_pred_rate, exposure, n_bins)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(tbl.index, tbl["obs_rate"], "o-", color="gray", label="Observed", lw=2)
    ax1.plot(tbl.index, tbl["pred_rate"], "s--", color="blue", label="Predicted", lw=2)
    ax1.set(xlabel="Decile (1=safest, 10=riskiest)", ylabel="Rate", title=title)
    ax1.legend(loc="upper left")

    # Exposure bar chart on twin axis
    ax2 = ax1.twinx()
    ax2.bar(tbl.index, tbl["exposure"], alpha=0.12, color="grey", label="Exposure")
    ax2.set_ylabel("Exposure")

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()
    plt.close(fig)

    # Summary diagnostics
    lift_ratio = tbl["pred_rate"].iloc[-1] / tbl["pred_rate"].iloc[0]
    crossings = int(
        np.sum(np.diff(np.sign(tbl["obs_rate"].values - tbl["pred_rate"].values)) != 0)
    )
    obs_monotonic = bool(np.all(np.diff(tbl["obs_rate"].values) >= 0))
    emit(
        f"  Lift ratio (decile 10 / decile 1): {lift_ratio:.2f}"
        f"  | Obs-Pred crossings: {crossings}"
        f"  | Observed monotonic: {obs_monotonic}"
    )
    return tbl


def double_lift_chart(
    y_true_rate: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    exposure: np.ndarray,
    label_a: str = "Model A",
    label_b: str = "Model B",
    n_bins: int = 10,
    title: str | None = None,
    filename: str | None = None,
) -> pd.DataFrame:
    """Double lift: bin by pred_A / pred_B, plot observed rate per bin.

    Increasing observed curve ⟹ A is right where it disagrees with B.
    Flat ⟹ models are equivalent. Decreasing ⟹ B is right.
    """
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    y_true_rate = np.asarray(y_true_rate, dtype=float)

    ratio = pred_a / np.fmax(pred_b, 1e-12)
    bin_labels = pd.qcut(ratio, q=n_bins, labels=False, duplicates="drop") + 1

    tmp = pd.DataFrame(
        {
            "bin": bin_labels,
            "exposure": exposure,
            "obs_total": y_true_rate * exposure,
            "pred_a_total": pred_a * exposure,
            "pred_b_total": pred_b * exposure,
        }
    )
    grp = tmp.groupby("bin").agg(
        exposure=("exposure", "sum"),
        obs_total=("obs_total", "sum"),
        pred_a_total=("pred_a_total", "sum"),
        pred_b_total=("pred_b_total", "sum"),
    )
    grp["obs_rate"] = grp["obs_total"] / grp["exposure"]
    grp["pred_a_rate"] = grp["pred_a_total"] / grp["exposure"]
    grp["pred_b_rate"] = grp["pred_b_total"] / grp["exposure"]

    portfolio_avg = (y_true_rate * exposure).sum() / exposure.sum()

    if title is None:
        title = f"Double lift: {label_a} vs {label_b}"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grp.index, grp["obs_rate"], "o-", color="gray", label="Observed", lw=2)
    ax.plot(
        grp.index, grp["pred_a_rate"], "s--", color="blue", label=f"Predicted ({label_a})", lw=1.5
    )
    ax.plot(
        grp.index, grp["pred_b_rate"], "^--", color="red", label=f"Predicted ({label_b})", lw=1.5
    )
    ax.axhline(portfolio_avg, ls=":", color="black", alpha=0.5, label="Portfolio avg")
    ax.set(
        xlabel=f"Decile of {label_a}/{label_b} ratio (1={label_b} higher, 10={label_a} higher)",
        ylabel="Rate",
        title=title,
    )
    ax.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()
    plt.close(fig)

    # Slope diagnostic via simple linear regression on bin medians
    x = grp.index.to_numpy(dtype=float)
    y = grp["obs_rate"].values
    slope = np.polyfit(x, y, 1)[0]
    if slope > 0:
        verdict = f"{label_a} captures signal that {label_b} misses (slope={slope:.4f})"
    elif slope < 0:
        verdict = f"{label_b} captures signal that {label_a} misses (slope={slope:.4f})"
    else:
        verdict = "Models are equivalent where they disagree"
    emit(f"  Double lift verdict: {verdict}")
    return grp





# %% [markdown]
# ## Tweedie power estimation (Pearson estimating equation)
#
# **Don't guess p. Estimate it.**
#
# We use statsmodels' iterative Pearson estimating equation: fit a Tweedie GLM at
# current p, solve the Pearson score equation for a new p, refit until convergence.
# This is fast and does not require density computation.


def estimate_tweedie_power_statsmodels(
    X: np.ndarray,
    y_total: np.ndarray,
    exposure: np.ndarray,
    p_init: float = 1.5,
    max_iter: int = 5,
    tol: float = 1e-4,
    low: float = 1.01,
    high: float = 1.99,
    max_samples: int = 50_000,
) -> tuple[float, list[float]]:
    """Estimate Tweedie power via statsmodels' Pearson estimating equation.

    Iterates: fit GLM at current p → solve Pearson score equation for p → refit.
    Uses the totals formulation with exposure (no rate/weight gymnastics).
    Subsamples to max_samples for speed on large datasets.

    Returns (p_hat, convergence_path).
    """
    # Subsample for speed : p estimation doesn't need all rows
    n = len(y_total)
    if n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_samples, replace=False)
        X, y_total, exposure = X[idx], y_total[idx], exposure[idx]

    X_with_const = sm.add_constant(X)
    path = [p_init]

    p_current = p_init
    for _ in range(max_iter):
        fam = TweedieFamily(var_power=p_current, link=sm.families.links.Log())
        model = sm.GLM(y_total, X_with_const, family=fam, exposure=exposure)
        res = model.fit(disp=False)
        p_new = model.estimate_tweedie_power(res.fittedvalues, low=low, high=high)
        path.append(p_new)
        if abs(p_new - p_current) < tol:
            break
        p_current = p_new

    return float(path[-1]), path





# %% [markdown]
# ## Metrics


# %%
def d2_explained(
    y_true_rate: np.ndarray,
    y_pred_rate: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute D^2 explained using Tweedie deviance.

    D^2 explained (GLM deviance analogue) computed from mean Tweedie deviance
    at the modeling power P; we use the same sample_weight for both terms.

    Parameters
    ----------
    y_true_rate : np.ndarray
        True rate values.
    y_pred_rate : np.ndarray
        Predicted rate values.
    sample_weight : np.ndarray | None = None
        Sample weights.

    Returns
    -------
    float
        The D^2 explained value.
    """
    sw = None if sample_weight is None else np.asarray(sample_weight)
    dev = mean_tweedie_deviance(y_true_rate, y_pred_rate, power=P, sample_weight=sw)
    y_bar = np.average(y_true_rate, weights=sw)
    dev_null = mean_tweedie_deviance(
        y_true_rate, np.full_like(y_true_rate, y_bar), power=P, sample_weight=sw
    )
    return 1.0 - (dev / dev_null if dev_null > 0 else np.nan)


def evaluate_models_table(
    df_test: pd.DataFrame,
    pred_dict: dict[str, np.ndarray],
    weights_for_eval: tuple[str, ...] = ("Exposure",),
) -> pd.DataFrame:
    """Build a table of exposure-weighted metrics for each pure premium model.

    Metrics are computed on rates:
    - MAE
    - MSE
    - mean Tweedie deviance at power P
    - D^2 explained

    All metrics are exposure-weighted, matching the actuarial pure premium
    rate formulation.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test data.
    pred_dict : dict[str, np.ndarray]
        Dictionary of model predictions.
    weights_for_eval : tuple[str, ...], optional
        Weighting schemes, by default ("Exposure",).

    Returns
    -------
    pd.DataFrame
        Table of metrics.
    """
    rows = []
    y_true_rate = df_test["PurePremium"].to_numpy()
    for wname in weights_for_eval:
        if wname == "Exposure":
            sw = rate_weights(df_test["Exposure"].to_numpy())
        else:
            raise ValueError("Unknown weight name")

        for model_name, y_pred_rate in pred_dict.items():
            mae = mean_absolute_error(y_true_rate, y_pred_rate, sample_weight=sw)
            mse = mean_squared_error(y_true_rate, y_pred_rate, sample_weight=sw)
            dev = mean_tweedie_deviance(
                y_true_rate, y_pred_rate, power=P, sample_weight=sw
            )
            d2 = d2_explained(y_true_rate, y_pred_rate, sample_weight=sw)
            rows.append(
                {
                    "eval_weight": wname,
                    "model": model_name,
                    "MAE_rate": mae,
                    "MSE_rate": mse,
                    f"MeanTweedieDev(p={P})": dev,
                    "D2_explained": d2,
                }
            )
    res = pd.DataFrame(rows).set_index(["eval_weight", "model"])
    return res


def d2_poisson_explained(
    y_true_rate: np.ndarray,
    y_pred_rate: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute D^2 explained using Poisson deviance.

    D^2 explained for Poisson deviance, evaluated on rates with exposure as sample_weight.

    Parameters
    ----------
    y_true_rate : np.ndarray
        True rate values.
    y_pred_rate : np.ndarray
        Predicted rate values.
    sample_weight : np.ndarray | None = None
        Sample weights.

    Returns
    -------
    float
        The D^2 explained value.
    """
    sw = None if sample_weight is None else np.asarray(sample_weight)
    dev = mean_poisson_deviance(y_true_rate, y_pred_rate, sample_weight=sw)
    y_bar = np.average(y_true_rate, weights=sw)
    dev_null = mean_poisson_deviance(
        y_true_rate, np.full_like(y_true_rate, y_bar), sample_weight=sw
    )
    return 1.0 - (dev / dev_null if dev_null > 0 else np.nan)


def evaluate_frequency_models_table(
    df_test: pd.DataFrame, pred_dict: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Build a table of metrics for frequency models.

    Build a tidy table of metrics for frequency models.
    Evaluation is always exposure-weighted.
    Metrics on rates: MAE, MSE, mean Poisson dev, and D^2 explained.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test data.
    pred_dict : dict[str, np.ndarray]
        Dictionary of model predictions.

    Returns
    -------
    pd.DataFrame
        Table of metrics.
    """
    rows = []
    y_true_rate = df_test["Frequency"].to_numpy()
    sw = df_test["Exposure"].to_numpy()

    for model_name, y_pred_rate in pred_dict.items():
        mae = mean_absolute_error(y_true_rate, y_pred_rate, sample_weight=sw)
        mse = mean_squared_error(y_true_rate, y_pred_rate, sample_weight=sw)
        dev = mean_poisson_deviance(y_true_rate, y_pred_rate, sample_weight=sw)
        d2 = d2_poisson_explained(y_true_rate, y_pred_rate, sample_weight=sw)
        rows.append(
            {
                "model": model_name,
                "MAE_freq": mae,
                "MSE_freq": mse,
                "MeanPoissonDev": dev,
                "D2_explained": d2,
            }
        )
    res = pd.DataFrame(rows).set_index("model")
    return res


# %% [markdown]
# ## Fitting: sklearn TweedieRegressor (rates)


# %%
def fit_sklearn_tweedie_rates(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[str, TweedieRegressor, np.ndarray, np.ndarray]:
    """Fit scikit-learn TweedieRegressor on pure premium rates.

    Actuarial pure premium formulation:
      label = PurePremium
      sample_weight = Exposure

    Returns (tag, model, y_pred_rate_train, y_pred_rate_test).
    """
    tag = "sklearn_tweedie_rate_w=exp"

    y_tr = df_train["PurePremium"].to_numpy(dtype=float)
    wtr = rate_weights(df_train["Exposure"].to_numpy(dtype=float))

    glm = TweedieRegressor(power=P, alpha=ALPHA, solver="newton-cholesky")
    glm.fit(X_tr, y_tr, sample_weight=wtr)

    yhat_tr_rate = glm.predict(X_tr)
    yhat_te_rate = glm.predict(X_te)

    return tag, glm, yhat_tr_rate, yhat_te_rate


# %% [markdown]
# ## Fitting: statsmodels Tweedie GLM (MLE, no regularization)


# %%
def fit_statsmodels_tweedie_rates(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[str, object, np.ndarray, np.ndarray]:
    """Fit statsmodels Tweedie GLM on pure premium rates.

    Actuarial pure premium formulation:
      label = PurePremium
      var_weights = Exposure
      family = Tweedie(P)
      link = log

    Returns (tag, result, y_pred_rate_train, y_pred_rate_test).
    """
    tag = "sm_tweedie_rate_w=exp"

    y_tr = df_train["PurePremium"].to_numpy(dtype=float)
    wtr = rate_weights(df_train["Exposure"].to_numpy(dtype=float))

    X_tr_c = sm.add_constant(X_tr)
    X_te_c = sm.add_constant(X_te)

    fam = TweedieFamily(var_power=P, link=sm.families.links.Log())
    model = sm.GLM(y_tr, X_tr_c, family=fam, var_weights=wtr)
    res = model.fit(disp=False)

    yhat_tr_rate = np.asarray(res.predict(X_tr_c))
    yhat_te_rate = np.asarray(res.predict(X_te_c))

    return tag, res, yhat_tr_rate, yhat_te_rate


# %% [markdown]
# ## Fitting: LightGBM Tweedie


# %%
def fit_lgb_rates(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[str, lgb.Booster | None, np.ndarray, np.ndarray]:
    """Fit LightGBM Tweedie on pure premium rates.

    Actuarial pure premium formulation:
      label = PurePremium
      weight = Exposure
      objective = tweedie

    Returns (tag, model, y_pred_rate_train, y_pred_rate_test).
    """
    y_tr = df_train["PurePremium"].to_numpy(dtype=float)
    y_te = df_test["PurePremium"].to_numpy(dtype=float)

    wtr = rate_weights(df_train["Exposure"].to_numpy(dtype=float))
    wte = rate_weights(df_test["Exposure"].to_numpy(dtype=float))

    tag = "lgb_tweedie_rate_w=exp"

    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=wtr)
    dvalid = lgb.Dataset(X_te, label=y_te, weight=wte, reference=dtrain)

    params = {
        "objective": "tweedie",
        "tweedie_variance_power": P,
        "learning_rate": LEARNING_RATE,
        "num_leaves": LGB_NUM_LEAVES,
        "min_data_in_leaf": LGB_MIN_DATA_IN_LEAF,
        "feature_fraction": LGB_FEATURE_FRACTION,
        "bagging_fraction": LGB_BAGGING_FRACTION,
        "bagging_freq": LGB_BAGGING_FREQ,
        "lambda_l2": LGB_LAMBDA_L2,
        "boost_from_average": not LGB_DISABLE_BOOST_FROM_AVERAGE,
        "verbose": -1,
        "seed": RANDOM_STATE,
    }

    gbm = lgb.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=NUM_BOOST_ROUND,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    yhat_tr_rate = np.asarray(gbm.predict(X_tr))
    yhat_te_rate = np.asarray(gbm.predict(X_te))

    return tag, gbm, yhat_tr_rate, yhat_te_rate


# %% [markdown]
# ## Fitting: Poisson Models for Frequency


# %%
def fit_sklearn_poisson_rates(
    X_tr: np.ndarray, X_te: np.ndarray, df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[str, PoissonRegressor, np.ndarray, np.ndarray]:
    """Fit scikit-learn PoissonRegressor.

    sklearn PoissonRegressor: rates + weights
      label = Frequency
      sample_weight = Exposure
    Returns (tag, model, y_pred_rate_train, y_pred_rate_test)

    Parameters
    ----------
    X_tr : np.ndarray
        Training features.
    X_te : np.ndarray
        Test features.
    df_train : pd.DataFrame
        Training data.
    df_test : pd.DataFrame
        Test data.

    Returns
    -------
    tuple[str, PoissonRegressor, np.ndarray, np.ndarray]
        Tag, model, train predictions, test predictions.
    """
    tag = "sklearn_poisson_rate_w=exp"
    wtr = df_train["Exposure"].to_numpy()
    y_tr = df_train["Frequency"].to_numpy()

    glm = PoissonRegressor(alpha=ALPHA, solver="newton-cholesky")
    glm.fit(X_tr, y_tr, sample_weight=wtr)
    yhat_tr_rate = glm.predict(X_tr)
    yhat_te_rate = glm.predict(X_te)
    return tag, glm, yhat_tr_rate, yhat_te_rate


def fit_lgb_poisson_offset_counts(
    X_tr: np.ndarray, X_te: np.ndarray, df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[str, lgb.Booster | None, np.ndarray, np.ndarray]:
    """Fit LightGBM Poisson with offset.

    LightGBM (Poisson) with log-exposure offset via init_score.
      - label = ClaimNb (counts)
      - init_score = log(Exposure)
      - objective = 'poisson'
      - predict() returns RATE per unit exposure (not counts)
    Returns: (tag, model, yhat_tr_rate, yhat_te_rate)

    Parameters
    ----------
    X_tr : np.ndarray
        Training features.
    X_te : np.ndarray
        Test features.
    df_train : pd.DataFrame
        Training data.
    df_test : pd.DataFrame
        Test data.

    Returns
    -------
    tuple[str, Optional[lgb.Booster], np.ndarray, np.ndarray]
        Tag, model, train predictions, test predictions.
    """
    tag = "lgb_poisson_offset"

    # Labels (counts)
    y_tr = df_train["ClaimNb"].to_numpy(dtype=float)
    y_te = df_test["ClaimNb"].to_numpy(dtype=float)

    exp_tr = df_train["Exposure"].to_numpy(dtype=float)
    exp_te = df_test["Exposure"].to_numpy(dtype=float)
    init_tr = log_exposure_offset(exp_tr)
    init_te = log_exposure_offset(exp_te)

    dtrain = lgb.Dataset(X_tr, label=y_tr, init_score=init_tr)
    dvalid = lgb.Dataset(X_te, label=y_te, init_score=init_te, reference=dtrain)

    params = {
        "objective": "poisson",
        "learning_rate": LEARNING_RATE,
        "num_leaves": LGB_NUM_LEAVES,
        "min_data_in_leaf": LGB_MIN_DATA_IN_LEAF,
        "feature_fraction": LGB_FEATURE_FRACTION,
        "bagging_fraction": LGB_BAGGING_FRACTION,
        "bagging_freq": LGB_BAGGING_FREQ,
        "lambda_l2": LGB_LAMBDA_L2,
        "boost_from_average": not LGB_DISABLE_BOOST_FROM_AVERAGE,
        "verbose": -1,
        "seed": RANDOM_STATE,
    }

    gbm = lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dvalid],
        num_boost_round=NUM_BOOST_ROUND,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    # As in the Tweedie offset example above, predict() is interpreted here as a
    # per-unit-exposure rate. Reconstruct totals by multiplying back by exposure.
    yhat_tr_rate = np.asarray(gbm.predict(X_tr))
    yhat_te_rate = np.asarray(gbm.predict(X_te))

    # Check: total count of claims predicted on test set
    yhat_te_tot = yhat_te_rate * exp_te
    print_summary_table(
        "LightGBM Poisson Offset Sanity Check",
        (
            ("Predicted total claim count", yhat_te_tot.sum()),
            ("Observed total claim count", y_te.sum()),
        ),
    )

    return tag, gbm, yhat_tr_rate, yhat_te_rate


def fit_lgb_poisson_rates_weights(
    X_tr: np.ndarray, X_te: np.ndarray, df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[str, lgb.Booster | None, np.ndarray, np.ndarray]:
    """Fit LightGBM Poisson with rates and weights.

    LightGBM: rates + weights
      label = Frequency
      weight = Exposure
      objective = poisson
    Returns (tag, model, yhat_tr_rate, yhat_te_rate)

    Parameters
    ----------
    X_tr : np.ndarray
        Training features.
    X_te : np.ndarray
        Test features.
    df_train : pd.DataFrame
        Training data.
    df_test : pd.DataFrame
        Test data.

    Returns
    -------
    tuple[str, Optional[lgb.Booster], np.ndarray, np.ndarray]
        Tag, model, train predictions, test predictions.
    """
    tag = "lgb_poisson_rate_w=exp"
    y_tr = df_train["Frequency"].to_numpy(dtype=float)
    y_te = df_test["Frequency"].to_numpy(dtype=float)
    wtr = rate_weights(df_train["Exposure"].to_numpy(dtype=float))
    wte = rate_weights(df_test["Exposure"].to_numpy(dtype=float))

    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=wtr)
    dvalid = lgb.Dataset(X_te, label=y_te, weight=wte, reference=dtrain)

    params = {
        "objective": "poisson",
        "learning_rate": LEARNING_RATE,
        "num_leaves": LGB_NUM_LEAVES,
        "min_data_in_leaf": LGB_MIN_DATA_IN_LEAF,
        "feature_fraction": LGB_FEATURE_FRACTION,
        "bagging_fraction": LGB_BAGGING_FRACTION,
        "bagging_freq": LGB_BAGGING_FREQ,
        "lambda_l2": LGB_LAMBDA_L2,
        "boost_from_average": not LGB_DISABLE_BOOST_FROM_AVERAGE,
        "verbose": -1,
        "seed": RANDOM_STATE,
    }
    gbm = lgb.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=NUM_BOOST_ROUND,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    yhat_tr_rate = np.asarray(gbm.predict(X_tr))
    yhat_te_rate = np.asarray(gbm.predict(X_te))
    return tag, gbm, yhat_tr_rate, yhat_te_rate


# %% [markdown]
# ## Fitting: XGBoost (Tweedie + Poisson)
#
# XGBoost has a native Tweedie objective (`reg:tweedie`) since 1.0 and a native
# Poisson objective (`count:poisson`). We use the same actuarial convention:
# `label = rate`, `weight = Exposure`. Hyperparameters mirror the LightGBM
# choices.


# %%
def fit_xgb_tweedie_rates(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[str, "xgb.Booster", np.ndarray, np.ndarray]:
    """Fit XGBoost Tweedie on pure premium rates."""
    y_tr = df_train["PurePremium"].to_numpy(dtype=float)
    y_te = df_test["PurePremium"].to_numpy(dtype=float)

    wtr = rate_weights(df_train["Exposure"].to_numpy(dtype=float))
    wte = rate_weights(df_test["Exposure"].to_numpy(dtype=float))

    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=wtr)
    dvalid = xgb.DMatrix(X_te, label=y_te, weight=wte)

    params = {
        "objective": "reg:tweedie",
        "tweedie_variance_power": P,
        "learning_rate": LEARNING_RATE,
        "max_depth": XGB_MAX_DEPTH,
        "min_child_weight": XGB_MIN_CHILD_WEIGHT,
        "subsample": XGB_SUBSAMPLE,
        "colsample_bytree": XGB_COLSAMPLE_BYTREE,
        "reg_lambda": XGB_REG_LAMBDA,
        "eval_metric": f"tweedie-nloglik@{P}",
        "tree_method": "hist",
        "verbosity": 0,
        "seed": RANDOM_STATE,
    }
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    yhat_tr_rate = np.asarray(booster.predict(xgb.DMatrix(X_tr)))
    yhat_te_rate = np.asarray(booster.predict(dvalid))
    return "xgb_tweedie_rate_w=exp", booster, yhat_tr_rate, yhat_te_rate


def fit_xgb_poisson_rates(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[str, "xgb.Booster", np.ndarray, np.ndarray]:
    """Fit XGBoost Poisson on frequency rates."""
    y_tr = df_train["Frequency"].to_numpy(dtype=float)
    y_te = df_test["Frequency"].to_numpy(dtype=float)

    wtr = rate_weights(df_train["Exposure"].to_numpy(dtype=float))
    wte = rate_weights(df_test["Exposure"].to_numpy(dtype=float))

    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=wtr)
    dvalid = xgb.DMatrix(X_te, label=y_te, weight=wte)

    params = {
        "objective": "count:poisson",
        "learning_rate": LEARNING_RATE,
        "max_depth": XGB_MAX_DEPTH,
        "min_child_weight": XGB_MIN_CHILD_WEIGHT,
        "subsample": XGB_SUBSAMPLE,
        "colsample_bytree": XGB_COLSAMPLE_BYTREE,
        "reg_lambda": XGB_REG_LAMBDA,
        "eval_metric": "poisson-nloglik",
        "tree_method": "hist",
        "verbosity": 0,
        "seed": RANDOM_STATE,
    }
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    yhat_tr_rate = np.asarray(booster.predict(xgb.DMatrix(X_tr)))
    yhat_te_rate = np.asarray(booster.predict(dvalid))
    return "xgb_poisson_rate_w=exp", booster, yhat_tr_rate, yhat_te_rate


# %% [markdown]
# ## Data loading and preprocessing

# %%
# Load and clean data
df = load_mtpl2(n_samples=N_SAMPLES)
df = basic_cleaning(df)

# Split BEFORE fitting transformer to avoid leakage (unsupervised transforms, but good hygiene)
df_train, df_test = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
)

# Build transformer on train, transform both
column_trans = build_column_transformer()
X_train = column_trans.fit_transform(df_train)
X_test = column_trans.transform(df_test)


# %% [markdown]
# ## Estimate Tweedie power from data
#
# `P_INIT = 1.35` is the starting guess. The Pearson estimating equation converges
# quickly (typically 3–5 iterations). After estimation, `P` is reassigned globally
# so all downstream fitting, weights, and metrics use the data-driven value.

# %%
p_hat_sm, p_path = estimate_tweedie_power_statsmodels(
    X_train,
    df_train["ClaimAmount"].to_numpy(dtype=float),
    df_train["Exposure"].to_numpy(dtype=float),
    p_init=P_INIT,
    max_iter=10,
)

# Reassign the global P to the estimated value
P = p_hat_sm

print_summary_table(
    "Tweedie Power Estimation",
    (
        ("p_init (starting guess)", P_INIT),
        ("p_hat (statsmodels Pearson)", P),
        ("Pearson iterations", len(p_path) - 1),
        ("Pearson convergence path", " -> ".join(f"{v:.4f}" for v in p_path)),
    ),
)

print_summary_table(
    "Run Configuration",
    (
        ("Train size", len(df_train)),
        ("Test size", len(df_test)),
        ("Tweedie power p (estimated)", P),
        ("Ridge alpha (sklearn)", ALPHA),
        ("LightGBM rounds", NUM_BOOST_ROUND),
        (
            "LightGBM base score",
            "boost_from_average=False (manual)"
            if LGB_DISABLE_BOOST_FROM_AVERAGE
            else "boost_from_average=True",
        ),
    ),
)

# %% [markdown]
# ## Pure Premium Model Fitting (Tweedie)
#
# We fit pure premium models using the actuarial rate formulation:
#
#     label = PurePremium = ClaimAmount / Exposure
#     sample_weight = Exposure
#
# This is the convention used throughout the tutorial for scikit-learn,
# statsmodels, and LightGBM rate models.
#
# The goal is not to prove algebraic equivalence between all possible encodings.
# That derivation belongs in the PDF notes. Here we focus on practical modelling:
#
# - consistent exposure handling,
# - Tweedie power estimation,
# - exposure-weighted metrics,
# - calibration,
# - lift and Lorenz diagnostics.

# %%
# --- sklearn: rates + weights ---
tag_sk_tw, sk_tw, sk_tw_tr, sk_tw_te = fit_sklearn_tweedie_rates(
    X_train, X_test, df_train, df_test
)

# --- statsmodels: rates + weights (unpenalized MLE) ---
tag_sm_rate, sm_rate_res, sm_rate_tr, sm_rate_te = fit_statsmodels_tweedie_rates(
    X_train, X_test, df_train, df_test
)

# --- GBM variants (if available) ---
gbm_pred_test: dict[str, np.ndarray] = {}
gbm_pred_train: dict[str, np.ndarray] = {}

if LGB_AVAILABLE:
    tag_lgb_tw, lgb_tw, lgb_tw_tr, lgb_tw_te = fit_lgb_rates(
        X_train, X_test, df_train, df_test
    )
    gbm_pred_test[tag_lgb_tw] = lgb_tw_te
    gbm_pred_train[tag_lgb_tw] = lgb_tw_tr

if XGB_AVAILABLE:
    tag_xgb_tw, xgb_tw, xgb_tw_tr, xgb_tw_te = fit_xgb_tweedie_rates(
        X_train, X_test, df_train, df_test
    )
    gbm_pred_test[tag_xgb_tw] = xgb_tw_te
    gbm_pred_train[tag_xgb_tw] = xgb_tw_tr

# Collect predictions on TEST (rates)
pred_rate_test: dict[str, np.ndarray] = {
    tag_sk_tw: sk_tw_te,
    tag_sm_rate: sm_rate_te,
}
pred_rate_test.update(gbm_pred_test)

# Collect predictions on TRAIN (rates) : needed for train-vs-test comparison
pred_rate_train: dict[str, np.ndarray] = {
    tag_sk_tw: sk_tw_tr,
    tag_sm_rate: sm_rate_tr,
}
pred_rate_train.update(gbm_pred_train)

# %% [markdown]
# ## Lorenz curves (TEST), exposure-weighted

# %%
y_true_rate = df_test["PurePremium"].to_numpy()
fig, ax = plt.subplots(figsize=(7, 7))
# models
exposure_test = df_test["Exposure"].to_numpy()
for label, y_pred_rate in pred_rate_test.items():
    cx, cy = lorenz_curve(y_true_rate, y_pred_rate, exposure_test)
    gini = 1 - 2 * auc(cx, cy)
    ax.plot(cx, cy, label=f"{label} (Gini={gini:.3f})")
# Oracle
cx, cy = lorenz_curve(y_true_rate, y_true_rate, exposure_test)
gini = 1 - 2 * auc(cx, cy)
ax.plot(cx, cy, linestyle="-.", label=f"Oracle (Gini={gini:.3f})")
# Random
ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
ax.set(
    title="Lorenz curves on TEST (exposure-weighted)",
    xlabel="Cumulative exposure (sorted by predicted risk, low→high)",
    ylabel="Cumulative claim amounts",
)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("lorenz_curve_tweedie.png")
plt.show()
plt.close(fig)
emit("Saved Tweedie Lorenz curve to lorenz_curve_tweedie.png", style="green")

# %% [markdown]
# ## Metrics tables (TEST) under exposure weighting

# %%
metrics_tbl = evaluate_models_table(
    df_test, pred_rate_test, weights_for_eval=("Exposure",)
)
print_grouped_metrics_tables(
    metrics_tbl.sort_index(), title_prefix="Tweedie Test Metrics"
)

# %% [markdown]
# ## Aggregate totals comparison (TEST)

# %%
y_true_tot = (df_test["PurePremium"].to_numpy() * exposure_test).sum()
agg_rows = [
    (tag, np.sum(exposure_test * pred)) for tag, pred in pred_rate_test.items()
]
agg_df = build_aggregate_comparison_df(
    observed_label="Observed totals",
    observed_value=y_true_tot,
    rows=tuple(agg_rows),
    value_col="sum_predicted_totals",
)
print_dataframe_table(
    agg_df,
    title="Aggregate Predicted Totals On TEST",
    caption="Absolute and relative error are measured against the observed total claim amount.",
)

# %% [markdown]
# ## Calibration by feature (TEST): pick 2 interpretable features

# %%
feature_list = ["DrivAge", "VehPower"]
for feat in feature_list:
    # --- Sklearn models comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # Get observed rates and distribution
    grp_obs = _get_aggregated_rates(df_test, feat, "Exposure", y_true_rate)

    # Plot observed rate as gray line without markers
    grp_obs["rate"].plot(
        style="-", color="gray", ax=ax, label="Observed", linewidth=1.5, alpha=0.8
    )

    # Plot predicted rates for each sklearn model with consistent colors and thicker lines
    model_styles_sklearn = {
        tag_sk_tw: ("-", "blue", "Predicted (sklearn)"),
    }

    for model_tag, (linestyle, color, label) in model_styles_sklearn.items():
        grp_pred = _get_aggregated_rates(
            df_test, feat, "Exposure", pred_rate_test[model_tag]
        )
        grp_pred["rate"].plot(
            style=linestyle, color=color, ax=ax, label=label, linewidth=3
        )

    # Add a shaded area for the feature's distribution
    y_max = ax.get_ylim()[1]
    x_values = (
        grp_obs.index.astype(float)
        if pd.api.types.is_numeric_dtype(grp_obs.index)
        else np.arange(len(grp_obs))
    )

    ax.fill_between(
        x_values,
        0,
        y_max * 0.5 * grp_obs["w"] / np.nanmax(grp_obs["w"]),
        alpha=0.1,
        color="grey",
        label=f"{feat} distribution",
    )

    ax.set(
        title=f"TEST: Calibration by {feat} (scikit-learn models)",
        xlabel=feat,
        ylabel="Pure premium (per exposure)",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"calibration_sklearn_{feat}.png")
    plt.show()
    plt.close(fig)
    emit(
        f"Saved sklearn calibration plot for {feat} to calibration_sklearn_{feat}.png",
        style="green",
    )

    # --- GBM models comparison ---
    gbm_tags = {
        tag: ("-", color, label)
        for tag, (color, label) in (
            ("lgb_tweedie_rate_w=exp", ("green", "Predicted (LGB)")),
            ("xgb_tweedie_rate_w=exp", ("orange", "Predicted (XGB)")),
        )
        if tag in pred_rate_test
    }
    if gbm_tags:
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot observed rate as gray line without markers
        grp_obs["rate"].plot(
            style="-", color="gray", ax=ax, label="Observed", linewidth=1.5, alpha=0.8
        )

        for model_tag, (linestyle, color, label) in gbm_tags.items():
            grp_pred = _get_aggregated_rates(
                df_test, feat, "Exposure", pred_rate_test[model_tag]
            )
            grp_pred["rate"].plot(
                style=linestyle, color=color, ax=ax, label=label, linewidth=3
            )

        # Add distribution shading
        ax.fill_between(
            x_values,
            0,
            y_max * 0.5 * grp_obs["w"] / np.nanmax(grp_obs["w"]),
            alpha=0.1,
            color="grey",
            label=f"{feat} distribution",
        )

        ax.set(
            title=f"TEST: Calibration by {feat} (GBM models)",
            xlabel=feat,
            ylabel="Pure premium (per exposure)",
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"calibration_gbm_{feat}.png")
        plt.show()
        plt.close(fig)
        emit(
            f"Saved GBM calibration plot for {feat} to calibration_gbm_{feat}.png",
            style="green",
        )

# %% [markdown]
# ---
# # FREQUENCY ANALYSIS (Poisson)
#
# The Poisson frequency case is the canonical exposure-rate example.
#
# Counts formulation:
#
#     E[ClaimNb_i] = Exposure_i * lambda_i
#
# Rate formulation:
#
#     Frequency_i = ClaimNb_i / Exposure_i
#     sample_weight = Exposure_i
#
# This section verifies that the same exposure convention used for pure premium
# also applies naturally to claim frequency.

# %% [markdown]
# ## Frequency Model Fitting (Poisson)

# %%
# --- sklearn: rates + weights ---
tag_sk_poi, sk_poi_model, sk_poi_tr, sk_poi_te = fit_sklearn_poisson_rates(
    X_train, X_test, df_train, df_test
)

# --- GBM variants (if available) ---
gbm_freq_pred_test: dict[str, np.ndarray] = {}
gbm_freq_pred_train: dict[str, np.ndarray] = {}

if LGB_AVAILABLE:
    tag_lgb_off, lgb_off_model, lgb_off_tr, lgb_off_te = fit_lgb_poisson_offset_counts(
        X_train, X_test, df_train, df_test
    )
    tag_lgb_w, lgb_w_model, lgb_w_tr, lgb_w_te = fit_lgb_poisson_rates_weights(
        X_train, X_test, df_train, df_test
    )
    gbm_freq_pred_test[tag_lgb_off] = lgb_off_te
    gbm_freq_pred_test[tag_lgb_w] = lgb_w_te
    gbm_freq_pred_train[tag_lgb_off] = lgb_off_tr
    gbm_freq_pred_train[tag_lgb_w] = lgb_w_tr

if XGB_AVAILABLE:
    tag_xgb_poi, xgb_poi, xgb_poi_tr, xgb_poi_te = fit_xgb_poisson_rates(
        X_train, X_test, df_train, df_test
    )
    gbm_freq_pred_test[tag_xgb_poi] = xgb_poi_te
    gbm_freq_pred_train[tag_xgb_poi] = xgb_poi_tr

# Collect all frequency predictions on TEST
pred_freq_test: dict[str, np.ndarray] = {tag_sk_poi: sk_poi_te}
pred_freq_test.update(gbm_freq_pred_test)

# Collect all frequency predictions on TRAIN
pred_freq_train: dict[str, np.ndarray] = {tag_sk_poi: sk_poi_tr}
pred_freq_train.update(gbm_freq_pred_train)

# %% [markdown]
# ## Frequency Metrics (TEST)

# %%
metrics_freq_tbl = evaluate_frequency_models_table(df_test, pred_freq_test)
print_dataframe_table(
    metrics_freq_tbl.sort_index().reset_index(),
    title="Poisson Frequency Test Metrics [Exposure]",
    caption="All frequency metrics are exposure-weighted, matching the Poisson rate formulation.",
)

# %% [markdown]
# ## Aggregate Claim Count Comparison (TEST)

# %%
exposure_test_freq = df_test["Exposure"].to_numpy()
y_true_counts = df_test["ClaimNb"].to_numpy().sum()

agg_rows_freq = []
for model_name, pred_rate in pred_freq_test.items():
    pred_counts = np.sum(exposure_test_freq * pred_rate)
    agg_rows_freq.append((model_name, pred_counts))

agg_df_freq = build_aggregate_comparison_df(
    observed_label="Observed counts",
    observed_value=y_true_counts,
    rows=tuple(agg_rows_freq),
    value_col="sum_predicted_counts",
)
print_dataframe_table(
    agg_df_freq,
    title="Aggregate Predicted Claim Counts On TEST",
    caption="Absolute and relative error are measured against the observed total claim count.",
)

# %% [markdown]
# ## Lorenz & Calibration Plots for Frequency (TEST)

# %%
y_true_freq = df_test["Frequency"].to_numpy()

# --- Lorenz Curve ---
fig, ax = plt.subplots(figsize=(7, 7))
for label, y_pred_freq in pred_freq_test.items():
    cx, cy = lorenz_curve(y_true_freq, y_pred_freq, exposure_test_freq)
    gini = 1 - 2 * auc(cx, cy)
    ax.plot(cx, cy, label=f"{label} (Gini={gini:.3f})")

# Oracle
cx, cy = lorenz_curve(y_true_freq, y_true_freq, exposure_test_freq)
gini = 1 - 2 * auc(cx, cy)
ax.plot(cx, cy, linestyle="-.", label=f"Oracle (Gini={gini:.3f})")

ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
ax.set(
    title="Lorenz curves for Frequency on TEST (exposure-weighted)",
    xlabel="Cumulative exposure (sorted by predicted frequency, low→high)",
    ylabel="Cumulative claim counts",
)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("lorenz_curve_frequency.png")
plt.show()
plt.close(fig)
emit("Saved Frequency Lorenz curve to lorenz_curve_frequency.png", style="green")

# --- Calibration Plot ---
feat = "DrivAge"  # Pick one feature for demonstration
fig, ax = plt.subplots(figsize=(8, 5))

# Get observed rates and distribution
grp_obs = _get_aggregated_rates(df_test, feat, "Exposure", y_true_freq)

# Plot observed rate as gray line without markers
grp_obs["rate"].plot(
    style="-", color="gray", ax=ax, label="Observed", linewidth=1.5, alpha=0.8
)

# Plot predicted rates for each model with consistent colors and thicker lines
model_styles_freq = {
    tag: ("-", color, label)
    for tag, (color, label) in (
        ("sklearn_poisson_rate_w=exp", ("blue", "Predicted (sklearn)")),
        ("lgb_poisson_offset", ("green", "Predicted (LGB offset)")),
        ("lgb_poisson_rate_w=exp", ("orange", "Predicted (LGB rates)")),
        ("xgb_poisson_rate_w=exp", ("red", "Predicted (XGB)")),
    )
    if tag in pred_freq_test
}

for model_tag, (linestyle, color, label) in model_styles_freq.items():
    grp_pred = _get_aggregated_rates(
        df_test, feat, "Exposure", pred_freq_test[model_tag]
    )
    grp_pred["rate"].plot(style=linestyle, color=color, ax=ax, label=label, linewidth=3)

# Add a shaded area for the feature's distribution
y_max = ax.get_ylim()[1]
x_values = (
    grp_obs.index.astype(float)
    if pd.api.types.is_numeric_dtype(grp_obs.index)
    else np.arange(len(grp_obs))
)

ax.fill_between(
    x_values,
    0,
    y_max * 0.5 * grp_obs["w"] / np.nanmax(grp_obs["w"]),
    alpha=0.1,
    color="grey",
    label=f"{feat} distribution",
)

ax.set(
    title=f"TEST: Frequency Calibration by {feat}",
    xlabel=feat,
    ylabel="Claim Frequency (per exposure)",
)
ax.legend()
plt.tight_layout()
plt.savefig("calibration_frequency_DrivAge.png")
plt.show()
plt.close(fig)
emit(
    "Saved Frequency calibration plot for DrivAge to calibration_frequency_DrivAge.png",
    style="green",
)

# %% [markdown]
# ---
# # MODEL VALIDATION TOOLKIT
#
# A model that scores well on deviance can still be catastrophically miscalibrated
# at the portfolio level. These 6 checks catch different failure modes:
#
# 1. **Ranking** : Lorenz curves + Gini on train & test (overfitting detection)
# 2. **Accuracy** : Deviance & $D^2$ on train vs test (side-by-side)
# 3. **Calibration** : Portfolio-level: sum(predicted) vs sum(observed)
# 4. **Lift charts** : 10-decile obs vs pred, discrimination range
# 5. **Train vs test summary** : all key metrics in one table
# 6. **Double lift** : head-to-head: who is right where they disagree?

# %% [markdown]
# ## 1. Ranking: Lorenz curves on TRAIN & TEST

# %%
exposure_train = df_train["Exposure"].to_numpy()
exposure_test_v = df_test["Exposure"].to_numpy()
y_true_rate_train = df_train["PurePremium"].to_numpy()
y_true_rate_test = df_test["PurePremium"].to_numpy()

# --- Compute Gini on both splits ---
gini_rows = []
for model_name in pred_rate_test:
    g_te = gini_coefficient(y_true_rate_test, pred_rate_test[model_name], exposure_test_v)
    g_tr = gini_coefficient(y_true_rate_train, pred_rate_train[model_name], exposure_train)
    gini_rows.append({"model": model_name, "Gini_train": g_tr, "Gini_test": g_te, "delta": g_tr - g_te})

gini_oracle_tr = gini_coefficient(y_true_rate_train, y_true_rate_train, exposure_train)
gini_oracle_te = gini_coefficient(y_true_rate_test, y_true_rate_test, exposure_test_v)
gini_rows.append({"model": "Oracle", "Gini_train": gini_oracle_tr, "Gini_test": gini_oracle_te, "delta": np.nan})

gini_df = pd.DataFrame(gini_rows)
print_dataframe_table(
    gini_df,
    title="Pure Premium : Gini (TRAIN vs TEST)",
    caption="delta = Gini_train - Gini_test. Large delta signals overfitting.",
)

# --- Lorenz plot: TRAIN ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for split_name, ax, y_true, preds, exp in [
    ("TRAIN", axes[0], y_true_rate_train, pred_rate_train, exposure_train),
    ("TEST", axes[1], y_true_rate_test, pred_rate_test, exposure_test_v),
]:
    lorenz_curves_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, y_pred in preds.items():
        cx, cy = lorenz_curve(y_true, y_pred, exp)
        g = 1 - 2 * auc(cx, cy)
        ax.plot(cx, cy, label=f"{label} ({g:.3f})")
        lorenz_curves_cache[label] = (cx, cy)
    # Oracle
    cx, cy = lorenz_curve(y_true, y_true, exp)
    g = 1 - 2 * auc(cx, cy)
    ax.plot(cx, cy, ls="-.", label=f"Oracle ({g:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.set(title=f"Lorenz : PP ({split_name})", xlabel="Cum. exposure", ylabel="Cum. claim amount")
    ax.legend(loc="lower right", fontsize=7)

plt.tight_layout()
plt.savefig("lorenz_pp_train_test.png")
plt.show()
plt.close(fig)
emit("Saved lorenz_pp_train_test.png", style="green")

# --- Dominance check on TEST ---
lorenz_test_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
for label, y_pred in pred_rate_test.items():
    lorenz_test_curves[label] = lorenz_curve(y_true_rate_test, y_pred, exposure_test_v)
dom_df = check_lorenz_dominance(lorenz_test_curves)
print_dataframe_table(dom_df, title="Lorenz Dominance (TEST, PP)", model_col="model_A")

# %% [markdown]
# ## 2. Accuracy: Deviance on TRAIN vs TEST (side-by-side)

# %%
metrics_train = evaluate_models_table(
    df_train, pred_rate_train, weights_for_eval=("Exposure",)
)
metrics_test_v = evaluate_models_table(
    df_test, pred_rate_test, weights_for_eval=("Exposure",)
)

# Merge train and test metrics
mt = metrics_train.reset_index()
mt = mt.rename(columns={c: c + "_train" for c in mt.columns if c not in ("eval_weight", "model")})
me = metrics_test_v.reset_index()
me = me.rename(columns={c: c + "_test" for c in me.columns if c not in ("eval_weight", "model")})
accuracy_df = mt.merge(me, on=["eval_weight", "model"])
print_dataframe_table(
    accuracy_df.drop(columns=["eval_weight"]),
    title="PP Accuracy: TRAIN vs TEST (exposure-weighted)",
    caption="Compare columns pairwise to detect overfitting.",
)

# %% [markdown]
# ## 3. Calibration: Portfolio-level totals

# %%
# --- Pure Premium calibration ---
obs_pp_total = df_test["ClaimAmount"].to_numpy().sum()
cal_pp_rows = []
for model_name, pred in pred_rate_test.items():
    pred_total = np.sum(exposure_test_v * pred)
    rel_err = 100.0 * (pred_total - obs_pp_total) / obs_pp_total
    cal_pp_rows.append({"model": model_name, "predicted_total": pred_total, "observed_total": obs_pp_total, "rel_error_%": rel_err})
cal_pp_df = pd.DataFrame(cal_pp_rows)
print_dataframe_table(
    cal_pp_df,
    title="Calibration : Pure Premium totals (TEST)",
    caption="rel_error_% should be < 1% for a well-calibrated model.",
)

# --- Frequency calibration ---
obs_count_total = float(df_test["ClaimNb"].to_numpy().sum())
cal_freq_rows = []
for model_name, pred in pred_freq_test.items():
    pred_total = np.sum(exposure_test_v * pred)
    rel_err = 100.0 * (pred_total - obs_count_total) / obs_count_total
    cal_freq_rows.append({"model": model_name, "predicted_count": pred_total, "observed_count": obs_count_total, "rel_error_%": rel_err})
cal_freq_df = pd.DataFrame(cal_freq_rows)
print_dataframe_table(
    cal_freq_df,
    title="Calibration : Claim counts (TEST)",
    caption="rel_error_% should be < 1% for a well-calibrated model.",
)

# %% [markdown]
# ## 4. Lift Charts (10 deciles, TRAIN & TEST)
#
# Bin policies by predicted risk into 10 deciles. In each bin:
#   obs_rate  = sum(ClaimAmount in bin) / sum(Exposure in bin)
#   pred_rate = sum(pred_PP * Exposure in bin) / sum(Exposure in bin)
#
# Key diagnostics:
# - Discrimination: lift ratio = max decile rate / min decile rate
# - No crossings: observed and predicted curves should not cross
# - Monotonicity: observed rate should increase across deciles

# %%
emit("\n[bold]Pure Premium : Lift Charts[/bold]")
for model_name in pred_rate_test:
    emit(f"\n  {model_name} : TEST:")
    lift_chart(
        y_true_rate_test, pred_rate_test[model_name], exposure_test_v,
        title=f"Lift chart TEST : {model_name} (PP)",
        filename=f"lift_pp_test_{model_name}.png",
    )
    emit(f"  {model_name} : TRAIN:")
    lift_chart(
        y_true_rate_train, pred_rate_train[model_name], exposure_train,
        title=f"Lift chart TRAIN : {model_name} (PP)",
        filename=f"lift_pp_train_{model_name}.png",
    )

emit("\n[bold]Frequency : Lift Charts[/bold]")
y_true_freq_train = df_train["Frequency"].to_numpy()
y_true_freq_test_v = df_test["Frequency"].to_numpy()
for model_name in pred_freq_test:
    emit(f"\n  {model_name} : TEST:")
    lift_chart(
        y_true_freq_test_v, pred_freq_test[model_name], exposure_test_v,
        title=f"Lift chart TEST : {model_name} (Freq)",
        filename=f"lift_freq_test_{model_name}.png",
    )
    emit(f"  {model_name} : TRAIN:")
    lift_chart(
        y_true_freq_train, pred_freq_train[model_name], exposure_train,
        title=f"Lift chart TRAIN : {model_name} (Freq)",
        filename=f"lift_freq_train_{model_name}.png",
    )

# %% [markdown]
# ## 5. Train vs Test summary table
#
# One compact table with Gini, deviance, and lift ratio on both splits.

# %%
summary_rows = []
for model_name in pred_rate_test:
    g_tr = gini_coefficient(y_true_rate_train, pred_rate_train[model_name], exposure_train)
    g_te = gini_coefficient(y_true_rate_test, pred_rate_test[model_name], exposure_test_v)
    sw_tr = df_train["Exposure"].to_numpy()
    sw_te = df_test["Exposure"].to_numpy()
    dev_tr = mean_tweedie_deviance(
        y_true_rate_train, pred_rate_train[model_name], power=P, sample_weight=sw_tr
    )
    dev_te = mean_tweedie_deviance(
        y_true_rate_test, pred_rate_test[model_name], power=P, sample_weight=sw_te
    )
    lift_tr = _compute_lift_table(y_true_rate_train, pred_rate_train[model_name], exposure_train)
    lift_te = _compute_lift_table(y_true_rate_test, pred_rate_test[model_name], exposure_test_v)
    lr_tr = lift_tr["pred_rate"].iloc[-1] / lift_tr["pred_rate"].iloc[0]
    lr_te = lift_te["pred_rate"].iloc[-1] / lift_te["pred_rate"].iloc[0]
    summary_rows.append(
        {
            "model": model_name,
            "Gini_train": g_tr,
            "Gini_test": g_te,
            "Gini_delta": g_tr - g_te,
            f"TweedieDev_train(p={P})": dev_tr,
            f"TweedieDev_test(p={P})": dev_te,
            "LiftRatio_train": lr_tr,
            "LiftRatio_test": lr_te,
        }
    )
summary_df = pd.DataFrame(summary_rows)
print_dataframe_table(
    summary_df,
    title="Train vs Test Summary (PP models)",
    caption="Gini_delta > 0.05 or LiftRatio_train >> LiftRatio_test suggests overfitting.",
)

# %% [markdown]
# ## 6. Double Lift Charts
#
# Compare two fitted models head-to-head by binning on the ratio of their
# predictions.
#
# - Increasing observed curve: model A captures signal that model B misses.
# - Flat observed curve: the models are similar where they disagree.
# - Decreasing observed curve: model B captures signal that model A misses.
#
# This section compares implementations or modelling choices, not alternative
# exposure-weight formulas.

# %%
emit("\n[bold]Double Lift : Pure Premium[/bold]")
if "lgb_tweedie_rate_w=exp" in pred_rate_test:
    double_lift_chart(
        y_true_rate_test,
        pred_rate_test[tag_sk_tw],
        pred_rate_test["lgb_tweedie_rate_w=exp"],
        exposure_test_v,
        label_a=tag_sk_tw,
        label_b="lgb_tweedie_rate_w=exp",
        title="Double lift (TEST): sklearn vs LightGBM Tweedie pure premium",
        filename="double_lift_pp_sklearn_vs_lgb.png",
    )

# LightGBM vs XGBoost head-to-head (when both available)
if (
    "lgb_tweedie_rate_w=exp" in pred_rate_test
    and "xgb_tweedie_rate_w=exp" in pred_rate_test
):
    double_lift_chart(
        y_true_rate_test,
        pred_rate_test["lgb_tweedie_rate_w=exp"],
        pred_rate_test["xgb_tweedie_rate_w=exp"],
        exposure_test_v,
        label_a="lgb_tweedie_rate_w=exp",
        label_b="xgb_tweedie_rate_w=exp",
        title="Double lift (TEST): LightGBM vs XGBoost Tweedie pure premium",
        filename="double_lift_pp_lgb_vs_xgb.png",
    )

double_lift_chart(
    y_true_rate_test,
    pred_rate_test[tag_sk_tw],
    pred_rate_test[tag_sm_rate],
    exposure_test_v,
    label_a=tag_sk_tw,
    label_b=tag_sm_rate,
    title="Double lift (TEST): sklearn vs statsmodels Tweedie pure premium",
    filename="double_lift_pp_sklearn_vs_sm.png",
)

# --- Frequency double lifts ---
emit("\n[bold]Double Lift : Frequency[/bold]")
if "lgb_poisson_offset" in pred_freq_test:
    double_lift_chart(
        y_true_freq_test_v,
        pred_freq_test[tag_sk_poi],
        pred_freq_test["lgb_poisson_offset"],
        exposure_test_v,
        label_a=tag_sk_poi,
        label_b="lgb_poisson_offset",
        title="Double lift (TEST): sklearn vs LightGBM Poisson frequency",
        filename="double_lift_freq_sklearn_vs_lgb.png",
    )
    double_lift_chart(
        y_true_freq_test_v,
        pred_freq_test["lgb_poisson_offset"],
        pred_freq_test["lgb_poisson_rate_w=exp"],
        exposure_test_v,
        label_a="lgb_poisson_offset",
        label_b="lgb_poisson_rate_w=exp",
        title="Double lift (TEST): LGB offset vs rate (Poisson frequency)",
        filename="double_lift_freq_lgb_offset_vs_rates.png",
    )

if (
    "lgb_poisson_rate_w=exp" in pred_freq_test
    and "xgb_poisson_rate_w=exp" in pred_freq_test
):
    double_lift_chart(
        y_true_freq_test_v,
        pred_freq_test["lgb_poisson_rate_w=exp"],
        pred_freq_test["xgb_poisson_rate_w=exp"],
        exposure_test_v,
        label_a="lgb_poisson_rate_w=exp",
        label_b="xgb_poisson_rate_w=exp",
        title="Double lift (TEST): LightGBM vs XGBoost Poisson frequency",
        filename="double_lift_freq_lgb_vs_xgb.png",
    )

# %%
