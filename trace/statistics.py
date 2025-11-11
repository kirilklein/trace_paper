"""
Risk Difference (RD) Statistics Module

This module handles the statistical pipeline for Risk Difference analysis.
It employs a hybrid inference approach to ensure robust statistics:

1. **Point Estimates & CIs (Risk Difference)**:
   Calculated using the Delta Method. We transform pooled logits back to
   probabilities, compute RD = p1 - p0, and estimate the CI based on the
   propagated standard error.
   (See: _compute_delta_method_rd_ci)

2. **Hypothesis Testing (P-values)**:
   Calculated using a t-test on the pooled logit scale (H0: logit(p1) = logit(p0)).
   This is generally more robust for probabilities near 0 or 1.
   (See: _compute_logit_difference_p_value)

Workflow:
  Inputs (Probs) -> Logit Transform -> Pooling
  -> Call (1) for RD and CIs
  -> Call (2) for P-Value
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.special import expit as inv_logit
from scipy.special import logit
from scipy.stats import norm
from scipy.stats import t as tdist

# -----------------------------
# Constants & Configuration
# -----------------------------
COL_EFFECT_1 = "effect_1"
COL_EFFECT_0 = "effect_0"
COL_E1_L = "effect_1_CI95_lower"
COL_E1_U = "effect_1_CI95_upper"
COL_E0_L = "effect_0_CI95_lower"
COL_E0_U = "effect_0_CI95_upper"


# -----------------------------
# Transformation Helpers
# -----------------------------
def compute_logit_se_from_ci(
    lo: Union[float, np.ndarray], hi: Union[float, np.ndarray], z: float = 1.96
) -> Union[float, np.ndarray]:
    """
    Derive the Standard Error (SE) on the logit scale from probability CIs.
    """
    lo, hi = np.asarray(lo), np.asarray(hi)
    return (logit(hi) - logit(lo)) / (2 * z)


def add_logit_arm_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment the DataFrame with logit-transformed point estimates and SEs
    for both treatment (1) and control (0) arms.
    """
    out = df.copy()

    # Arm 1 (Treatment)
    out["eta1"] = logit(out[COL_EFFECT_1])
    out["effect_1_logit_CI95_lower"] = logit(out[COL_E1_L])
    out["effect_1_logit_CI95_upper"] = logit(out[COL_E1_U])
    out["se_eta1"] = compute_logit_se_from_ci(out[COL_E1_L], out[COL_E1_U])

    # Arm 0 (Control)
    out["eta0"] = logit(out[COL_EFFECT_0])
    out["effect_0_logit_CI95_lower"] = logit(out[COL_E0_L])
    out["effect_0_logit_CI95_upper"] = logit(out[COL_E0_U])
    out["se_eta0"] = compute_logit_se_from_ci(out[COL_E0_L], out[COL_E0_U])

    return out


def pool_arm_logits(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    eta_col: str,
    se_col: str,
    out_prefix: str,
    pooling: Literal["simple_mean"] = "simple_mean",
) -> pd.DataFrame:
    """
    Pools logit estimates within groups.
    """
    group_cols_list = [group_cols] if isinstance(group_cols, str) else list(group_cols)

    def _agg_wrapper(g: pd.DataFrame) -> pd.Series:
        cleaned = (
            g.replace([np.inf, -np.inf], np.nan).dropna(subset=[eta_col, se_col]).copy()
        )
        m = len(cleaned)

        result = {
            out_prefix: np.nan,
            f"{out_prefix}_se": np.nan,
            f"{out_prefix}_tau2": np.nan,
            "n_runs_used": m,  # This 'm' is what we'll use
            "df": np.nan,  # 'df' column is no longer used but harmless
            "method_used": pooling,
        }

        if m == 0:
            return pd.Series(result)

        yi = cleaned[eta_col].astype(float).values

        if pooling == "simple_mean":
            theta = float(np.mean(yi))
            if m >= 2:
                se_hat = float(np.std(yi, ddof=1)) / np.sqrt(m)
                df_s = m - 1
            else:
                se_hat = np.nan
                df_s = np.nan

            # We still set 'df' for diagnostic purposes, but our
            # p-value function will use 'n_runs_used' (m) directly.
            result.update({out_prefix: theta, f"{out_prefix}_se": se_hat, "df": df_s})

        else:
            raise ValueError(f"Invalid pooling method: {pooling}")

        return pd.Series(result)

    return (
        df.groupby(group_cols_list, as_index=False)
        .apply(_agg_wrapper, include_groups=False)
        .reset_index(drop=True)
    )


# -----------------------------
# Inference: Logit Difference (P-Value)
# -----------------------------
def _compute_logit_difference_p_value(
    eta1: np.ndarray,
    se1: np.ndarray,
    eta0: np.ndarray,
    se0: np.ndarray,
    num_runs: np.ndarray,
    ci_level: float = 0.95,
) -> Dict[str, np.ndarray]:
    """
    Performs a statistical test on the difference of logits.

    This function handles TWO cases:
    1. Pooled data (t-test): If num_runs >= 2, it performs a t-test
       with (num_runs - 1) degrees of freedom.
    2. Per-row data (z-test): If num_runs is NaN or < 2, it performs a
       z-test (Normal distribution approximation).

    Logic:
      H0: logit(p1) - logit(p0) = 0
      stat = (eta1 - eta0) / sqrt(se1^2 + se0^2)
    """
    alpha = 1.0 - ci_level

    # 1. Compute Difference, SE, and the Test Statistic
    diff_logit = eta1 - eta0
    diff_se = np.sqrt(se1**2 + se0**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        # This is a 't' or 'z' statistic depending on the df
        stat = np.divide(
            diff_logit, diff_se, out=np.full_like(diff_logit, np.nan), where=diff_se > 0
        )

    # 2. Determine Degrees of Freedom explicitly
    df_logit = num_runs - 1

    # 3. Allocate output arrays
    p_vals = np.full_like(stat, np.nan)
    crit_val = np.full_like(stat, np.nan)  # Critical value for CI

    # 4. Calculate P-values based on which test to use

    # --- Case A: Use t-distribution ---
    # True only if df_logit is a valid number (>= 1)
    use_t_dist_mask = np.isfinite(df_logit) & (df_logit >= 1) & np.isfinite(stat)
    if use_t_dist_mask.any():
        p_vals[use_t_dist_mask] = 2.0 * (
            1.0 - tdist.cdf(np.abs(stat[use_t_dist_mask]), df=df_logit[use_t_dist_mask])
        )
        crit_val[use_t_dist_mask] = tdist.ppf(
            1 - alpha / 2, df=df_logit[use_t_dist_mask]
        )

    # --- Case B: Use Normal distribution (z-test fallback) ---
    # True for rows that are NOT Case A (e.g., per-row with NaN df)
    use_z_dist_mask = (~use_t_dist_mask) & np.isfinite(stat)
    if use_z_dist_mask.any():
        p_vals[use_z_dist_mask] = 2.0 * (1.0 - norm.cdf(np.abs(stat[use_z_dist_mask])))
        crit_val[use_z_dist_mask] = norm.ppf(1 - alpha / 2)

    # 5. Calculate CIs on the logit scale
    ci_lo = diff_logit - crit_val * diff_se
    ci_hi = diff_logit + crit_val * diff_se

    # 6. Store Results
    return {
        "eta_diff": diff_logit,
        "se_eta_diff": diff_se,
        "df_logit": df_logit,
        # Return statistic as 'z' to match the column name in main.py
        "z": stat,
        "p_value": p_vals,
        "eta_diff_CI95_lower": ci_lo,
        "eta_diff_CI95_upper": ci_hi,
    }


# -----------------------------
# Inference: Delta Method (RD Estimates & CI)
# -----------------------------
def _compute_delta_method_rd_ci(
    eta1: np.ndarray,
    se1: np.ndarray,
    eta0: np.ndarray,
    se0: np.ndarray,
    z_crit: float = 1.96,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Calculates Risk Difference (RD) and its SE/CI using the Delta Method.
    This function *only* computes the effect size and its confidence,
    not the p-value.
    """
    # 1. Transform back to probability scale
    p1 = inv_logit(eta1)
    p0 = inv_logit(eta0)
    rd = p1 - p0

    # 2. Delta Method Variance Propagation
    deriv1 = p1 * (1 - p1)
    deriv0 = p0 * (1 - p0)

    var_rd = (deriv1**2 * se1**2) + (deriv0**2 * se0**2)
    se_rd = np.sqrt(var_rd)

    # 3. Calculate Confidence Interval for RD
    rd_lo = rd - z_crit * se_rd
    rd_hi = rd + z_crit * se_rd

    # 4. Diagnostics
    if verbose:
        n_tiny_se = np.sum(se_rd < 1e-6)
        if n_tiny_se > 0:
            print(f"  [DIAGNOSTIC] WARNING: SE_RD < 1e-6 count: {n_tiny_se}")

    return {
        "RD": rd,
        "SE_RD": se_rd,
        "RD_CI95_lower": rd_lo,
        "RD_CI95_upper": rd_hi,
        "p1_hat": p1,
        "p0_hat": p0,
    }


# -----------------------------
# Main Orchestrator
# -----------------------------
def compute_rd_pvalues(
    df: pd.DataFrame,
    group_cols: Optional[Union[str, List[str]]] = None,
    pooling_method: str = "inverse_variance_arms",  # retained for legacy compat
    arm_pooling: Literal["fixed_effect", "random_effects_hksj"] = "random_effects_hksj",
    arm_pooling_rho: Optional[float] = None,
    arm_weight_col: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Main entry point for RD calculation and inference.

    Pipeline:
    1. Adds logit transformations to the raw DataFrame.
    2. Calls _compute_delta_method_rd_ci for RD and CIs.
    3. Calls _compute_logit_difference_p_value for p-value and z-score.
    """

    # 1. Prepare Data (Logit Transform)
    df_logit = add_logit_arm_metrics(df)

    # 2. Path A: No Grouping (Per-row analysis)
    if not group_cols:
        if verbose:
            print("  [INFO] Computing per-row RD statistics.")

        # Get RD and CIs
        rd_stats = _compute_delta_method_rd_ci(
            eta1=df_logit["eta1"].values,
            se1=df_logit["se_eta1"].values,
            eta0=df_logit["eta0"].values,
            se0=df_logit["se_eta0"].values,
            verbose=verbose,
        )

        # Get P-value
        # For per-row, we pass NaN for num_runs to trigger z-test
        nan_array = np.full(len(df_logit), np.nan)
        p_value_stats = _compute_logit_difference_p_value(
            eta1=df_logit["eta1"].values,
            se1=df_logit["se_eta1"].values,
            eta0=df_logit["eta0"].values,
            se0=df_logit["se_eta0"].values,
            num_runs=nan_array,
        )

        out = df_logit.copy()
        for k, v in rd_stats.items():
            out[k] = v
        for k, v in p_value_stats.items():
            out[k] = v
        return out

    # 3. Path B: Pooling Arms
    if verbose:
        print(f"  [INFO] Pooling arms by {group_cols} using {arm_pooling}.")

    # Pool Treatment Arm
    arm1 = pool_arm_logits(
        df_logit, group_cols, "eta1", "se_eta1", "eta1_pooled", pooling=arm_pooling
    ).rename(
        columns={
            "df": "eta1_df",
            "method_used": "eta1_method",
            "eta1_pooled_tau2": "eta1_tau2",
        }
    )

    # Pool Control Arm
    arm0 = pool_arm_logits(
        df_logit, group_cols, "eta0", "se_eta0", "eta0_pooled", pooling=arm_pooling
    ).rename(
        columns={
            "df": "eta0_df",
            "method_used": "eta0_method",
            "eta0_pooled_tau2": "eta0_tau2",
        }
    )

    # Merge Arms
    group_cols_list = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    pooled = pd.merge(arm1, arm0, on=group_cols_list, how="inner")

    # Track sample sizes
    pooled.rename(
        columns={"n_runs_used_x": "n_runs_arm1", "n_runs_used_y": "n_runs_arm0"},
        inplace=True,
    )
    # This 'n_runs_shared' is the 'm' we need
    pooled["n_runs_shared"] = np.minimum(
        pooled["n_runs_arm1"].fillna(0), pooled["n_runs_arm0"].fillna(0)
    ).astype(int)

    # 4. Calculate Statistics (in two clean steps)

    # A. Get RD and CIs from pooled estimates
    rd_stats = _compute_delta_method_rd_ci(
        eta1=pooled["eta1_pooled"].values,
        se1=pooled["eta1_pooled_se"].values,
        eta0=pooled["eta0_pooled"].values,
        se0=pooled["eta0_pooled_se"].values,
        verbose=verbose,
    )
    for k, v in rd_stats.items():
        pooled[k] = v

    # B. Get P-value from pooled estimates
    p_value_stats = _compute_logit_difference_p_value(
        eta1=pooled["eta1_pooled"].values,
        se1=pooled["eta1_pooled_se"].values,
        eta0=pooled["eta0_pooled"].values,
        se0=pooled["eta0_pooled_se"].values,
        num_runs=pooled["n_runs_shared"].values,
    )
    for k, v in p_value_stats.items():
        pooled[k] = v

    return pooled
