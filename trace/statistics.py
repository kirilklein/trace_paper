"""
Risk Difference (RD) P-value Calculation Module

This module provides functions for computing risk differences and their associated
p-values using inverse-variance pooling and the delta method. The approach works
by transforming arm-level probabilities to the logit scale, pooling them across
runs, and then computing risk differences with appropriate standard errors.

Key Features:
- Logit-scale transformations for probability estimates
- Inverse-variance weighted pooling across multiple runs
- Delta method for risk difference standard errors
- Per-run and pooled p-value calculations

Typical workflow:
1. Start with arm-level probability estimates and their 95% CIs
2. Transform to logit scale and compute standard errors
3. (Optional) Pool estimates across runs using inverse-variance weighting
4. Compute risk differences and p-values via the delta method
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Literal
from scipy.stats import t as tdist, norm

# -----------------------------
# Column configuration
# -----------------------------
COL_EFFECT_1 = "effect_1"
COL_EFFECT_0 = "effect_0"
COL_E1_L = "effect_1_CI95_lower"
COL_E1_U = "effect_1_CI95_upper"
COL_E0_L = "effect_0_CI95_lower"
COL_E0_U = "effect_0_CI95_upper"


# -----------------------------
# Utilities
# -----------------------------
def _clip_prob(p: Union[float, np.ndarray], eps: float = 1e-8) -> np.ndarray:
    """
    Clip probabilities to [eps, 1-eps] to avoid log(0) or log(1).

    Parameters
    ----------
    p : float or np.ndarray
        Probability or array of probabilities
    eps : float, optional
        Small value to clip probabilities away from 0 and 1 (default: 1e-8)

    Returns
    -------
    np.ndarray
        Clipped probabilities
    """
    return np.clip(np.asarray(p, dtype=float), eps, 1 - eps)


def logit(p: Union[float, np.ndarray]) -> np.ndarray:
    """
    Compute the logit transformation: log(p / (1-p)).

    Parameters
    ----------
    p : float or np.ndarray
        Probability or array of probabilities

    Returns
    -------
    np.ndarray
        Logit-transformed values
    """
    p = _clip_prob(p)
    return np.log(p / (1 - p))


def inv_logit(x: Union[float, np.ndarray]) -> np.ndarray:
    """
    Compute the inverse logit (sigmoid) transformation: 1 / (1 + exp(-x)).

    Parameters
    ----------
    x : float or np.ndarray
        Value or array of values on the logit scale

    Returns
    -------
    np.ndarray
        Probabilities on [0,1]
    """
    x = np.asarray(x)
    return 1.0 / (1.0 + np.exp(-x))


def se_from_prob_ci_on_logit(
    lo: Union[float, np.ndarray], hi: Union[float, np.ndarray], z: float = 1.96
) -> np.ndarray:
    """
    Convert probability-scale CI [lo, hi] to a logit-scale SE via CI width.

    Assumes the CI on the probability scale corresponds to point estimate ± z*SE
    on the logit scale.

    Parameters
    ----------
    lo : float or np.ndarray
        Lower bound of 95% CI on probability scale
    hi : float or np.ndarray
        Upper bound of 95% CI on probability scale
    z : float, optional
        Z-score for the confidence level (default: 1.96 for 95% CI)

    Returns
    -------
    np.ndarray
        Standard error on the logit scale
    """
    lo, hi = np.asarray(lo), np.asarray(hi)
    return (logit(hi) - logit(lo)) / (2 * z)


# -----------------------------
# Step A: add arm-level logit metrics (per row)
# -----------------------------
def add_logit_arm_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-row logit metrics for each arm.

    Computes logit transformations and standard errors for both treatment
    arms based on probability estimates and their confidence intervals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: effect_1, effect_0, effect_1_CI95_lower,
        effect_1_CI95_upper, effect_0_CI95_lower, effect_0_CI95_upper

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with added columns:
        - eta1, se_eta1: logit(effect_1) and its SE
        - eta0, se_eta0: logit(effect_0) and its SE
        - effect_1_logit_CI95_lower/upper: CI bounds on logit scale
        - effect_0_logit_CI95_lower/upper: CI bounds on logit scale
    """
    out = df.copy()

    # Arm 1 (treatment)
    out["eta1"] = logit(out[COL_EFFECT_1])
    out["effect_1_logit_CI95_lower"] = logit(out[COL_E1_L])
    out["effect_1_logit_CI95_upper"] = logit(out[COL_E1_U])
    out["se_eta1"] = se_from_prob_ci_on_logit(out[COL_E1_L], out[COL_E1_U])

    # Arm 0 (control)
    out["eta0"] = logit(out[COL_EFFECT_0])
    out["effect_0_logit_CI95_lower"] = logit(out[COL_E0_L])
    out["effect_0_logit_CI95_upper"] = logit(out[COL_E0_U])
    out["se_eta0"] = se_from_prob_ci_on_logit(out[COL_E0_L], out[COL_E0_U])

    return out


# -----------------------------
# Step B: inverse-variance pool arm logits (optional)
# -----------------------------
def pool_arm_logits(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    eta_col: str,
    se_col: str,
    out_prefix: str,
    pooling: Literal["fixed_effect", "random_effects_hksj"] = "fixed_effect",
    ci: float = 0.95,
) -> pd.DataFrame:
    """
    Pool a logit parameter across runs via inverse-variance weighting.

    For each group, computes a weighted average of the logit estimates where
    weights are proportional to 1/SE^2.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with logit estimates and their standard errors
    group_cols : str or list of str
        Column(s) to group by (e.g., ["method", "outcome"])
    eta_col : str
        Column name containing logit estimates
    se_col : str
        Column name containing standard errors
    out_prefix : str
        Prefix for output columns

    Returns
    -------
    pd.DataFrame
        One row per group with columns:
        - {out_prefix}: pooled logit estimate
        - {out_prefix}_se: pooled SE
        - n_runs_used: number of runs included
    """

    if isinstance(group_cols, str):
        group_cols_list: List[str] = [group_cols]
    else:
        group_cols_list = list(group_cols)

    alpha = 1.0 - ci

    def _agg(g: pd.DataFrame) -> pd.Series:
        cleaned = (
            g.replace([np.inf, -np.inf], np.nan).dropna(subset=[eta_col, se_col]).copy()
        )

        m = int(cleaned.shape[0])
        if m == 0:
            return pd.Series(
                {
                    out_prefix: np.nan,
                    f"{out_prefix}_se": np.nan,
                    f"{out_prefix}_tau2": np.nan,
                    "n_runs_used": 0,
                    "df": np.nan,
                    "method_used": pooling,
                }
            )

        yi = cleaned[eta_col].astype(float).values
        sei = cleaned[se_col].astype(float).values
        vi = sei**2

        if pooling == "fixed_effect" or m == 1:
            w = np.divide(1.0, vi, out=np.zeros_like(vi), where=vi > 0)
            weight_sum = np.sum(w)
            if weight_sum <= 0:
                eta_hat = np.nan
                se_hat = np.nan
            else:
                eta_hat = np.sum(w * yi) / weight_sum
                se_hat = np.sqrt(1.0 / weight_sum)
            return pd.Series(
                {
                    out_prefix: eta_hat,
                    f"{out_prefix}_se": se_hat,
                    f"{out_prefix}_tau2": 0.0,
                    "n_runs_used": m,
                    "df": np.nan if m < 2 else m - 1,
                    "method_used": (
                        "fixed_effect" if pooling == "fixed_effect" else "single_run"
                    ),
                }
            )

        # Random-effects DL + HKSJ
        wi = np.divide(1.0, vi, out=np.zeros_like(vi), where=vi > 0)
        weight_sum = np.sum(wi)
        if weight_sum <= 0:
            return pd.Series(
                {
                    out_prefix: np.nan,
                    f"{out_prefix}_se": np.nan,
                    f"{out_prefix}_tau2": np.nan,
                    "n_runs_used": m,
                    "df": np.nan,
                    "method_used": "DL_RE_HKSJ",
                }
            )

        theta_fe = np.sum(wi * yi) / weight_sum
        Q = np.sum(wi * (yi - theta_fe) ** 2)
        df_q = m - 1
        c = weight_sum - (np.sum(wi**2) / weight_sum)
        tau2 = max((Q - df_q) / c, 0.0) if c > 0 else 0.0

        w_star = np.divide(1.0, vi + tau2, out=np.zeros_like(vi), where=(vi + tau2) > 0)
        w_star_sum = np.sum(w_star)
        if w_star_sum <= 0:
            eta_hat = np.nan
            se_hk = np.nan
        else:
            eta_hat = np.sum(w_star * yi) / w_star_sum
            num = np.sum(w_star * (yi - eta_hat) ** 2)
            den = (m - 1) * w_star_sum
            if den > 0:
                se_hk = np.sqrt(num / den)
            else:
                se_hk = np.sqrt(1.0 / w_star_sum) if w_star_sum > 0 else np.nan

        df_hk = m - 1 if m >= 2 else np.nan

        return pd.Series(
            {
                out_prefix: eta_hat,
                f"{out_prefix}_se": se_hk,
                f"{out_prefix}_tau2": tau2,
                "n_runs_used": m,
                "df": df_hk,
                "method_used": "DL_RE_HKSJ",
            }
        )

    return (
        df.groupby(group_cols_list, as_index=False)
        .apply(_agg, include_groups=False)
        .reset_index(drop=True)
    )


# -----------------------------
# Helper: logit-scale inference for pooled arms
# -----------------------------
def _add_logit_scale_inference(df: pd.DataFrame, ci: float = 0.95) -> pd.DataFrame:
    """Augment pooled arm logits with logit-scale difference inference."""

    if df.empty:
        return df

    alpha = 1.0 - ci

    eta1 = df["eta1_pooled"].astype(float).values
    eta0 = df["eta0_pooled"].astype(float).values
    se1 = df["eta1_pooled_se"].astype(float).values
    se0 = df["eta0_pooled_se"].astype(float).values

    eta_diff = eta1 - eta0
    se_diff = np.sqrt(se1**2 + se0**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.divide(
            eta_diff,
            se_diff,
            out=np.full_like(eta_diff, np.nan),
            where=se_diff > 0,
        )

    df1 = df.get("eta1_df", pd.Series(np.nan, index=df.index)).astype(float).values
    df0 = df.get("eta0_df", pd.Series(np.nan, index=df.index)).astype(float).values
    df_logit = np.fmin(df1, df0)

    p_vals = np.full_like(t_stat, np.nan)
    tcrit = np.full_like(t_stat, np.nan)

    for i, (t_val, df_val, s_val) in enumerate(zip(t_stat, df_logit, se_diff)):
        if not np.isfinite(t_val):
            continue

        if np.isfinite(df_val) and df_val >= 1:
            p_vals[i] = 2 * (1 - tdist.cdf(abs(t_val), df=df_val))
            tcrit[i] = tdist.ppf(1 - alpha / 2, df=df_val)
        else:
            p_vals[i] = 2 * (1 - norm.cdf(abs(t_val)))
            tcrit[i] = norm.ppf(1 - alpha / 2)

        if not np.isfinite(s_val) or s_val <= 0:
            tcrit[i] = np.nan

    lo = eta_diff - tcrit * se_diff
    hi = eta_diff + tcrit * se_diff

    df = df.copy()
    df["eta_diff"] = eta_diff
    df["se_eta_diff"] = se_diff
    df["df_logit"] = df_logit
    df["t_logit"] = t_stat
    df["p_value_logit"] = p_vals
    df["eta_diff_CI95_lower"] = lo
    df["eta_diff_CI95_upper"] = hi

    # Preserve delta-method diagnostics but promote logit-scale inference
    df.rename(columns={"p_value": "p_value_delta", "z": "z_delta"}, inplace=True)
    df["p_value"] = df["p_value_logit"]
    df["z"] = df["t_logit"]

    return df


# -----------------------------
# Step C: RD inference from arm logits (delta method)
# -----------------------------
def rd_inference_from_arm_logits(
    eta1: np.ndarray,
    se1: np.ndarray,
    eta0: np.ndarray,
    se0: np.ndarray,
    z_crit: float = 1.96,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute risk difference inference from arm logits using the delta method.

    Transforms logit-scale estimates back to probability scale, computes
    risk difference (RD = p1 - p0), and uses the delta method to obtain
    the standard error of RD. Then computes Wald z-statistic, p-value,
    and confidence interval.

    Parameters
    ----------
    eta1 : np.ndarray
        Logit estimates for arm 1 (treatment)
    se1 : np.ndarray
        Standard errors for eta1
    eta0 : np.ndarray
        Logit estimates for arm 0 (control)
    se0 : np.ndarray
        Standard errors for eta0
    z_crit : float, optional
        Critical z-value for confidence intervals (default: 1.96 for 95% CI)
    verbose : bool, optional
        If True, print diagnostic warnings for extreme values (default: False)

    Returns
    -------
    dict
        Dictionary with keys:
        - RD: risk difference (p1 - p0)
        - SE_RD: standard error of RD
        - z: Wald z-statistic
        - p_value: two-sided p-value
        - RD_CI95_lower: lower bound of 95% CI for RD
        - RD_CI95_upper: upper bound of 95% CI for RD
        - p1_hat: estimated probability for arm 1
        - p0_hat: estimated probability for arm 0

    Notes
    -----
    Assumes no covariance between arm estimates (independence assumption).
    The delta method variance is: Var(RD) = (dp/deta)^2 * Var(eta)
    where dp/deta = p*(1-p) for the inverse logit transformation.
    """
    # DIAGNOSTIC: Check for extreme input values
    if verbose:
        n_tiny_se1 = np.sum(se1 < 1e-6)
        n_tiny_se0 = np.sum(se0 < 1e-6)
        if n_tiny_se1 > 0 or n_tiny_se0 > 0:
            print(
                f"  WARNING: Found very small SEs: se1 < 1e-6: {n_tiny_se1}, "
                f"se0 < 1e-6: {n_tiny_se0}"
            )
            if n_tiny_se1 > 0:
                print(f"    Min se1: {np.min(se1):.2e}")
            if n_tiny_se0 > 0:
                print(f"    Min se0: {np.min(se0):.2e}")

    # Transform from logit scale to probability scale
    p1 = inv_logit(eta1)
    p0 = inv_logit(eta0)

    # Compute risk difference on probability scale
    rd = p1 - p0

    # DIAGNOSTIC: Check for extreme probabilities
    if verbose:
        n_extreme_p1 = np.sum((p1 < 0.001) | (p1 > 0.999))
        n_extreme_p0 = np.sum((p0 < 0.001) | (p0 > 0.999))
        if n_extreme_p1 > 0 or n_extreme_p0 > 0:
            print(
                f"  WARNING: Extreme probabilities: p1 < 0.001 or > 0.999: {n_extreme_p1}, "
                f"p0 < 0.001 or > 0.999: {n_extreme_p0}"
            )

    # Gaussian error propagation (delta method) for SE(RD)
    #
    # We have: RD = p1 - p0 where p_i = inv_logit(eta_i) = 1/(1 + exp(-eta_i))
    #
    # By the delta method:
    #   Var(p_i) = (dp_i/deta_i)² * Var(eta_i)
    #
    # The derivative of inv_logit is:
    #   dp_i/deta_i = p_i * (1 - p_i)
    #
    # Therefore:
    #   Var(p_i) = [p_i * (1 - p_i)]² * SE(eta_i)²
    #
    # Since RD = p1 - p0 and assuming independence between arms:
    #   Var(RD) = Var(p1) + Var(p0)
    #           = [p1*(1-p1)]² * SE(eta1)² + [p0*(1-p0)]² * SE(eta0)²
    #
    var_rd = (p1 * (1 - p1)) ** 2 * (se1**2) + (p0 * (1 - p0)) ** 2 * (se0**2)
    se_rd = np.sqrt(var_rd)

    # DIAGNOSTIC: Check for very small SE_RD
    if verbose:
        n_tiny_se_rd = np.sum(se_rd < 1e-6)
        if n_tiny_se_rd > 0:
            print(f"  WARNING: Very small SE_RD < 1e-6: {n_tiny_se_rd}")
            print(f"    Min SE_RD: {np.min(se_rd):.2e}")
            # Show which component is driving it
            var_term1 = (p1 * (1 - p1)) ** 2 * (se1**2)
            var_term0 = (p0 * (1 - p0)) ** 2 * (se0**2)
            print(f"    Variance components:")
            print(
                f"      Term 1 (arm 1): min={np.min(var_term1):.2e}, median={np.median(var_term1):.2e}"
            )
            print(
                f"      Term 0 (arm 0): min={np.min(var_term0):.2e}, median={np.median(var_term0):.2e}"
            )

    z = rd / se_rd
    pval = 2 * (1 - norm.cdf(np.abs(z)))

    # DIAGNOSTIC: Check for extreme z-statistics
    if verbose:
        n_extreme_z = np.sum(np.abs(z) > 30)
        n_very_extreme_z = np.sum(np.abs(z) > 50)
        if n_extreme_z > 0:
            print(
                f"  WARNING: Extreme z-statistics: |z| > 30: {n_extreme_z}, "
                f"|z| > 50: {n_very_extreme_z}"
            )
            print(f"    Max |z|: {np.max(np.abs(z)):.2f}")

    rd_lo = rd - z_crit * se_rd
    rd_hi = rd + z_crit * se_rd

    return {
        "RD": rd,
        "SE_RD": se_rd,
        "z": z,
        "p_value": pval,
        "RD_CI95_lower": rd_lo,
        "RD_CI95_upper": rd_hi,
        "p1_hat": p1,
        "p0_hat": p0,
    }


# -----------------------------
# Risk ratio inference
# -----------------------------
def rr_inference_from_arm_logits(
    eta1: np.ndarray,
    se1: np.ndarray,
    eta0: np.ndarray,
    se0: np.ndarray,
    z_crit: float = 1.96,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute risk ratio inference from arm logits using the delta method.

    Parameters
    ----------
    eta1, se1 : np.ndarray
        Logit estimates and standard errors for the treatment arm.
    eta0, se0 : np.ndarray
        Logit estimates and standard errors for the control arm.
    z_crit : float, optional
        Critical value for confidence intervals (default: 1.96 for 95% CI).
    verbose : bool, optional
        If True, emit diagnostic information for extreme values.

    Returns
    -------
    dict
        Keys include:
        - RR: risk ratio on the probability scale (p1 / p0)
        - log_RR: natural log of the risk ratio
        - SE_log_RR: standard error of log_RR
        - SE_RR: delta-method SE of RR
        - z: Wald z-statistic for log_RR (null hypothesis log_RR = 0)
        - p_value: two-sided p-value
        - log_RR_CI95_lower / log_RR_CI95_upper: confidence interval bounds on log scale
        - RR_CI95_lower / RR_CI95_upper: confidence interval bounds on RR scale
        - p1_hat / p0_hat: probabilities for each arm
    """

    p1 = inv_logit(eta1)
    p0 = inv_logit(eta0)

    if verbose:
        n_extreme = np.sum((p1 < 1e-4) | (p0 < 1e-4))
        if n_extreme > 0:
            print(
                f"  [RR] Very small probabilities detected (p < 1e-4) in {n_extreme} rows."
            )

    # Log risk ratio and its variance via delta method
    log_rr = np.log(p1) - np.log(p0)
    d_log_rr_d_eta1 = 1 - p1
    d_log_rr_d_eta0 = -(1 - p0)

    var_log_rr = (d_log_rr_d_eta1**2) * (se1**2) + (d_log_rr_d_eta0**2) * (se0**2)
    se_log_rr = np.sqrt(var_log_rr)

    with np.errstate(divide="ignore", invalid="ignore"):
        rr = np.exp(log_rr)
        se_rr = rr * se_log_rr

    # Handle potential zero standard errors gracefully
    z = np.divide(
        log_rr, se_log_rr, out=np.full_like(log_rr, np.nan), where=se_log_rr > 0
    )
    pval = 2 * (1 - norm.cdf(np.abs(z)))

    log_rr_lo = log_rr - z_crit * se_log_rr
    log_rr_hi = log_rr + z_crit * se_log_rr
    rr_lo = np.exp(log_rr_lo)
    rr_hi = np.exp(log_rr_hi)

    if verbose:
        n_tiny_se = np.sum(se_log_rr < 1e-6)
        if n_tiny_se > 0:
            print(f"  [RR] Very small SE_log_RR < 1e-6: {n_tiny_se}")

    return {
        "RR": rr,
        "log_RR": log_rr,
        "SE_log_RR": se_log_rr,
        "SE_RR": se_rr,
        "z": z,
        "p_value": pval,
        "log_RR_CI95_lower": log_rr_lo,
        "log_RR_CI95_upper": log_rr_hi,
        "RR_CI95_lower": rr_lo,
        "RR_CI95_upper": rr_hi,
        "p1_hat": p1,
        "p0_hat": p0,
    }


# -----------------------------
# Orchestrator
# -----------------------------
def compute_rd_pvalues(
    df: pd.DataFrame,
    group_cols: Optional[Union[str, List[str]]] = None,
    pooling_method: str = "inverse_variance_arms",
    arm_pooling: Literal["fixed_effect", "random_effects_hksj"] = "random_effects_hksj",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute risk difference (RD) p-values either per row or pooled across runs.

    Main orchestrator function that handles the complete workflow:
    1. Add logit metrics to input data
    2. Either compute per-row RDs or pool by group then compute RDs
    3. Return DataFrame with RD estimates and inference

    Arm-level pooling (selected via ``arm_pooling``)
    -----------------------------------------------
    The pipeline pools on the logit scale per arm, then performs a Wald
    t-test on the pooled logit difference and maps back to probabilities.

    ``arm_pooling`` controls the arm-level pooling strategy:
      - ``\"random_effects_hksj\"`` (default): DerSimonian–Laird RE with
        Hartung–Knapp–Sidik–Jonkman standard errors
      - ``\"fixed_effect\"``: inverse-variance fixed-effect pooling

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with arm-level probability estimates and CIs.
        Required columns: effect_1, effect_0, effect_1_CI95_lower,
        effect_1_CI95_upper, effect_0_CI95_lower, effect_0_CI95_upper
    group_cols : str, list of str, or None, optional
        If None, compute per-row RDs. If specified, pool within each group
        (e.g., ["method", "outcome"]) before computing RDs.
    pooling_method : str, optional
        Legacy switch for alternative RD-level poolers. Left for compatibility;
        the pipeline uses only arm-level pooling by default.
    arm_pooling : {\"fixed_effect\", \"random_effects_hksj\"}, optional
        Arm-level pooling method to use on the logit scale (default: RE+HKSJ).
    verbose : bool, optional
        If True, print diagnostic information during computation (default: False)

    Returns
    -------
    pd.DataFrame
        If group_cols is None: original DataFrame with added RD columns
        If group_cols specified: one row per group with pooled RD estimates

        Added columns include: RD, SE_RD, z, p_value, RD_CI95_lower,
        RD_CI95_upper, p1_hat, p0_hat, eta_diff, se_eta_diff,
        p_value_logit, eta_diff_CI95_lower, eta_diff_CI95_upper (logit
        diagnostics only when pooling across groups).

    Notes
    -----
    Arm-level pooling preserves arm structure and propagates uncertainty
    correctly through transformations. RD-level poolers are retained for
    sensitivity only and are not exposed via CLI.

    Examples
    --------
    >>> # Recommended: Pool arms separately
    >>> df_pooled = compute_rd_pvalues(df, group_cols=['method', 'outcome'])

    >>> # Sensitivity: Compare with Rubin's rules
    >>> df_rubin = compute_rd_pvalues(df, group_cols=['method', 'outcome'],
    ...                                pooling_method='rubins_rules')
    """
    if verbose:
        print(
            f"\n  [DIAGNOSTIC] Starting compute_rd_pvalues with method '{pooling_method}'..."
        )

    # For Rubin's rules and random effects, we need per-run RDs first
    if group_cols and pooling_method in ["rubins_rules", "random_effects_dl"]:
        if verbose:
            print(f"  [DIAGNOSTIC] Computing per-run RDs first...")

        # Compute per-run RDs (no grouping)
        df_per_run = compute_rd_pvalues(
            df, group_cols=None, pooling_method="inverse_variance_arms", verbose=False
        )

        # Now pool the RDs using the specified method
        if pooling_method == "rubins_rules":
            if verbose:
                print(f"  [DIAGNOSTIC] Applying Rubin's rules pooling...")
            return combine_rubins_rules(df_per_run, group_cols=group_cols)
        elif pooling_method == "random_effects_dl":
            if verbose:
                print(
                    f"  [DIAGNOSTIC] Applying DerSimonian-Laird random effects pooling..."
                )
            return combine_random_effects_DL(df_per_run, group_cols=group_cols)

    # Original inverse-variance-on-arms method
    dfl = add_logit_arm_metrics(df)

    if verbose:
        print(f"  [DIAGNOSTIC] After logit transformation:")
        print(
            f"    se_eta1: min={dfl['se_eta1'].min():.2e}, "
            f"median={dfl['se_eta1'].median():.2e}, max={dfl['se_eta1'].max():.2e}"
        )
        print(
            f"    se_eta0: min={dfl['se_eta0'].min():.2e}, "
            f"median={dfl['se_eta0'].median():.2e}, max={dfl['se_eta0'].max():.2e}"
        )

    if not group_cols:
        # Per-row computation
        if verbose:
            print(f"  [DIAGNOSTIC] Computing per-row RDs...")
        res = rd_inference_from_arm_logits(
            eta1=dfl["eta1"].values,
            se1=dfl["se_eta1"].values,
            eta0=dfl["eta0"].values,
            se0=dfl["se_eta0"].values,
            verbose=verbose,
        )
        out = dfl.copy()
        for k, v in res.items():
            out[k] = v
        return out

    # Pooled over groups using inverse-variance on arms
    if verbose:
        print(f"  [DIAGNOSTIC] Pooling over groups: {group_cols}")

    group_cols_list = [group_cols] if isinstance(group_cols, str) else list(group_cols)

    arm1 = pool_arm_logits(
        dfl,
        group_cols,
        "eta1",
        "se_eta1",
        "eta1_pooled",
        pooling=arm_pooling,
    ).rename(
        columns={
            "df": "eta1_df",
            "method_used": "eta1_method",
            "eta1_pooled_tau2": "eta1_tau2",
        }
    )
    arm0 = pool_arm_logits(
        dfl,
        group_cols,
        "eta0",
        "se_eta0",
        "eta0_pooled",
        pooling=arm_pooling,
    ).rename(
        columns={
            "df": "eta0_df",
            "method_used": "eta0_method",
            "eta0_pooled_tau2": "eta0_tau2",
        }
    )

    pooled = pd.merge(arm1, arm0, on=group_cols_list, how="inner")
    pooled.rename(
        columns={
            "n_runs_used_x": "n_runs_arm1",
            "n_runs_used_y": "n_runs_arm0",
        },
        inplace=True,
    )
    pooled["n_runs_shared"] = np.minimum(
        pooled["n_runs_arm1"].fillna(0).astype(int),
        pooled["n_runs_arm0"].fillna(0).astype(int),
    )

    if verbose:
        print(f"  [DIAGNOSTIC] After pooling:")
        print(
            f"    eta1_pooled_se: min={pooled['eta1_pooled_se'].min():.2e}, "
            f"median={pooled['eta1_pooled_se'].median():.2e}, "
            f"max={pooled['eta1_pooled_se'].max():.2e}"
        )
        print(
            f"    eta0_pooled_se: min={pooled['eta0_pooled_se'].min():.2e}, "
            f"median={pooled['eta0_pooled_se'].median():.2e}, "
            f"max={pooled['eta0_pooled_se'].max():.2e}"
        )
        print(f"  [DIAGNOSTIC] Computing RDs from pooled estimates...")

    res = rd_inference_from_arm_logits(
        eta1=pooled["eta1_pooled"].values,
        se1=pooled["eta1_pooled_se"].values,
        eta0=pooled["eta0_pooled"].values,
        se0=pooled["eta0_pooled_se"].values,
        verbose=verbose,
    )
    for k, v in res.items():
        pooled[k] = v

    # Preserve delta-method diagnostics but perform inference on the logit scale
    pooled = _add_logit_scale_inference(pooled, ci=0.95)
    return pooled


# -----------------------------
# Risk ratio orchestrator
# -----------------------------
def compute_rr_from_arm_estimates(
    df: pd.DataFrame,
    group_cols: Optional[Union[str, List[str]]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute risk ratio (RR) statistics from arm-level probability estimates.

    The workflow mirrors ``compute_rd_pvalues`` but produces risk ratio metrics,
    including log risk ratios, standard errors, and Wald-based inference.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with columns required by ``add_logit_arm_metrics`` (effect_1,
        effect_0, and their 95% CIs) plus optional grouping columns.
    group_cols : str or list of str, optional
        Columns used to aggregate runs before computing risk ratios. If None,
        the computation is done per input row.
    verbose : bool, optional
        If True, emit diagnostic information during the calculation.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the original columns plus risk ratio metrics.
    """

    dfl = add_logit_arm_metrics(df)

    if verbose:
        print(
            "  [RR] After logit transformation: se_eta1 min/median/max = "
            f"{dfl['se_eta1'].min():.2e} / {dfl['se_eta1'].median():.2e} / {dfl['se_eta1'].max():.2e}"
        )
        print(
            "  [RR] After logit transformation: se_eta0 min/median/max = "
            f"{dfl['se_eta0'].min():.2e} / {dfl['se_eta0'].median():.2e} / {dfl['se_eta0'].max():.2e}"
        )

    if not group_cols:
        if verbose:
            print("  [RR] Computing per-row risk ratios...")
        res = rr_inference_from_arm_logits(
            eta1=dfl["eta1"].values,
            se1=dfl["se_eta1"].values,
            eta0=dfl["eta0"].values,
            se0=dfl["se_eta0"].values,
            verbose=verbose,
        )
        out = dfl.copy()
        for k, v in res.items():
            out[k] = v
        return out

    if verbose:
        print(f"  [RR] Pooling arm logits over groups: {group_cols}")

    arm1 = pool_arm_logits(dfl, group_cols, "eta1", "se_eta1", "eta1_pooled")
    arm0 = pool_arm_logits(dfl, group_cols, "eta0", "se_eta0", "eta0_pooled")

    pooled = pd.merge(arm1, arm0, on=group_cols, how="inner")

    res = rr_inference_from_arm_logits(
        eta1=pooled["eta1_pooled"].values,
        se1=pooled["eta1_pooled_se"].values,
        eta0=pooled["eta0_pooled"].values,
        se0=pooled["eta0_pooled_se"].values,
        verbose=verbose,
    )

    for k, v in res.items():
        pooled[k] = v

    return pooled


# Import tdist for Rubin's rules
from scipy.stats import t as tdist

# Expect df with columns: method, outcome, run_id, RD, SE_RD
# You can feed either per-run results or per-run-per-method-per-outcome rows.


# -----------------------------
# Rubin's rules combiner
# -----------------------------
def combine_rubins_rules(
    df: pd.DataFrame, group_cols=("method", "outcome"), zcrit=1.96
) -> pd.DataFrame:
    """
    Pool risk differences across runs using Rubin's rules.

    This method is adapted from multiple imputation literature (Rubin, 1987).
    It pools pre-computed RD estimates by combining within-run and between-run
    variance components.

    Mathematical Details
    --------------------
    For m runs with estimates theta_i and variances U_i:

    1. Pooled estimate: theta_bar = mean(theta_i)
    2. Within-run variance: W_bar = mean(U_i)
    3. Between-run variance: B = var(theta_i)
    4. Total variance: T = W_bar + (1 + 1/m) * B
    5. Standard error: SE = sqrt(T)
    6. Degrees of freedom (Barnard-Rubin):
       df = (m-1) * [1 + W_bar / ((1 + 1/m) * B)]²
    7. Inference uses t-distribution with df

    When to Use
    -----------
    - Runs represent multiple imputations or bootstrap samples
    - You want to account for between-run variability
    - Sensitivity analysis to compare with arm-level pooling

    NOT recommended when:
    - Arm-level data is available (use inverse_variance_arms instead)
    - Runs should be weighted by precision (use random_effects_dl)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with per-run RD estimates
        Required columns: RD, SE_RD, plus columns in group_cols
    group_cols : tuple or list of str, optional
        Columns to group by (default: ("method", "outcome"))
    zcrit : float, optional
        Critical value for confidence intervals (default: 1.96 for 95% CI)

    Returns
    -------
    pd.DataFrame
        One row per group with columns:
        - RD: pooled risk difference
        - SE_RD: pooled standard error
        - z: test statistic (actually t-statistic if m >= 2)
        - p_value: two-sided p-value
        - RD_CI95_lower, RD_CI95_upper: confidence interval bounds
        - df: degrees of freedom
        - m_runs: number of runs pooled
        - method_used: "rubin" or "rubin_single"

    References
    ----------
    Rubin, D. B. (1987). Multiple Imputation for Nonresponse in Surveys.
    Wiley, New York.

    Barnard, J., & Rubin, D. B. (1999). Small-sample degrees of freedom
    with multiple imputation. Biometrika, 86(4), 948-955.
    """

    def _agg(g):
        thetas = g["RD"].astype(float).values
        U = (g["SE_RD"].astype(float).values) ** 2
        m = len(g)
        if m < 2:
            # Fall back to the single run
            theta_bar = thetas.mean()
            T = U.mean() if m == 1 else np.nan
            se = np.sqrt(T)
            z = theta_bar / se
            p = 2 * (1 - norm.cdf(abs(z)))
            return pd.Series(
                dict(
                    RD=theta_bar,
                    SE_RD=se,
                    df=np.nan,
                    z=np.nan,
                    p_value=p,
                    RD_CI95_lower=theta_bar - zcrit * se,
                    RD_CI95_upper=theta_bar + zcrit * se,
                    m_runs=m,
                    method_used="rubin_single",
                )
            )

        theta_bar = thetas.mean()
        Wbar = U.mean()
        B = thetas.var(ddof=1)
        T = Wbar + (1 + 1 / m) * B
        se = np.sqrt(T)

        # Barnard–Rubin df
        if B == 0:
            # no between-run variability -> use z approx
            tstat = theta_bar / se
            p = 2 * (1 - norm.cdf(abs(tstat)))
            df = np.inf
        else:
            df = (m - 1) * (1 + Wbar / ((1 + 1 / m) * B)) ** 2
            tstat = theta_bar / se
            p = 2 * (1 - tdist.cdf(abs(tstat), df=df))

        return pd.Series(
            dict(
                RD=theta_bar,
                SE_RD=se,
                df=df,
                z=tstat,
                p_value=p,
                RD_CI95_lower=theta_bar - zcrit * se,
                RD_CI95_upper=theta_bar + zcrit * se,
                m_runs=m,
                method_used="rubin",
            )
        )

    out = (
        df.groupby(list(group_cols), as_index=False).apply(_agg).reset_index(drop=True)
    )
    return out


def combine_rr_random_effects_HKSJ(
    df: pd.DataFrame,
    group_cols=("method", "outcome"),
    ci: float = 0.95,
) -> pd.DataFrame:
    """
    Random-effects meta-analysis of log risk ratios using HKSJ adjustment.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``log_RR`` and ``SE_log_RR`` in addition to the
        grouping columns.
    group_cols : tuple/list of str, optional
        Columns to group by when pooling (default: ("method", "outcome")).
    ci : float, optional
        Confidence level for the output intervals (default: 0.95).

    Returns
    -------
    pd.DataFrame
        DataFrame with log- and probability-scale risk ratio summaries.
    """

    required = set(group_cols) | {"log_RR", "SE_log_RR"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "combine_rr_random_effects_HKSJ missing required columns: "
            + ", ".join(sorted(missing))
        )

    rr_pool = (
        df[list(group_cols) + ["log_RR", "SE_log_RR"]]
        .rename(columns={"log_RR": "RD", "SE_log_RR": "SE_RD"})
        .copy()
    )

    combined = combine_random_effects_HKSJ(rr_pool, group_cols=group_cols, ci=ci)

    combined = combined.rename(
        columns={
            "RD": "log_RR",
            "SE_RD": "SE_log_RR",
            "RD_CI95_lower": "log_RR_CI95_lower",
            "RD_CI95_upper": "log_RR_CI95_upper",
        }
    )

    combined["RR"] = np.exp(combined["log_RR"])
    combined["SE_RR"] = combined["RR"] * combined["SE_log_RR"]
    combined["RR_CI95_lower"] = np.exp(combined["log_RR_CI95_lower"])
    combined["RR_CI95_upper"] = np.exp(combined["log_RR_CI95_upper"])

    return combined


# -----------------------------
# Random-effects (DerSimonian–Laird) combiner
# -----------------------------
def combine_random_effects_DL(
    df: pd.DataFrame, group_cols=("method", "outcome"), zcrit=1.96
) -> pd.DataFrame:
    """
    Pool risk differences using DerSimonian-Laird random effects meta-analysis.

    This classic random effects method estimates between-study heterogeneity
    and uses it to adjust the pooling weights. It's commonly used in meta-analysis
    when heterogeneity is expected across studies/runs.

    Mathematical Details
    --------------------
    For k runs with estimates y_i and variances v_i:

    1. Fixed-effect pooled estimate (for Q statistic):
       theta_FE = sum(w_i * y_i) / sum(w_i), where w_i = 1/v_i

    2. Cochran's Q statistic:
       Q = sum(w_i * (y_i - theta_FE)²)

    3. Heterogeneity variance (DerSimonian-Laird estimator):
       tau² = max(0, (Q - (k-1)) / C)
       where C = sum(w_i) - sum(w_i²) / sum(w_i)

    4. Random effects weights:
       w_i* = 1 / (v_i + tau²)

    5. Random effects pooled estimate:
       theta_RE = sum(w_i* * y_i) / sum(w_i*)
       SE_RE = sqrt(1 / sum(w_i*))

    6. Inference uses normal distribution (z-test)

    When to Use
    -----------
    - Substantial heterogeneity expected across runs
    - Runs come from different populations/settings
    - Conservative inference desired (wider CIs than fixed effects)
    - Sensitivity analysis to assess impact of heterogeneity

    NOT recommended when:
    - Arm-level data is available (use inverse_variance_arms instead)
    - No heterogeneity expected (use fixed effects or Rubin's rules)
    - Very few runs (tau² estimate unreliable with k < 5)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with per-run RD estimates
        Required columns: RD, SE_RD, plus columns in group_cols
    group_cols : tuple or list of str, optional
        Columns to group by (default: ("method", "outcome"))
    zcrit : float, optional
        Critical value for confidence intervals (default: 1.96 for 95% CI)

    Returns
    -------
    pd.DataFrame
        One row per group with columns:
        - RD: pooled risk difference
        - SE_RD: pooled standard error
        - z: z-statistic
        - p_value: two-sided p-value
        - RD_CI95_lower, RD_CI95_upper: confidence interval bounds
        - tau2: estimated between-run heterogeneity variance
        - m_runs: number of runs pooled
        - method_used: "DL_RE"

    References
    ----------
    DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials.
    Controlled Clinical Trials, 7(3), 177-188.

    Notes
    -----
    When tau² = 0 (no heterogeneity), this reduces to fixed-effect
    inverse-variance pooling. Large tau² indicates substantial heterogeneity,
    suggesting results may not be directly comparable across runs.
    """

    def _agg(g):
        yi = g["RD"].astype(float).values
        vi = (g["SE_RD"].astype(float).values) ** 2
        wi = 1.0 / vi

        # Fixed-effect pooled (for Q)
        theta_FE = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - theta_FE) ** 2)
        k = len(yi)
        df_Q = max(k - 1, 1)
        c = np.sum(wi) - (np.sum(wi**2) / np.sum(wi))
        tau2 = max((Q - df_Q) / c, 0.0) if c > 0 else 0.0

        w_star = 1.0 / (vi + tau2)
        theta_RE = np.sum(w_star * yi) / np.sum(w_star)
        se_RE = np.sqrt(1.0 / np.sum(w_star))
        z = theta_RE / se_RE
        p = 2 * (1 - norm.cdf(abs(z)))

        return pd.Series(
            dict(
                RD=theta_RE,
                SE_RD=se_RE,
                tau2=tau2,
                z=z,
                p_value=p,
                RD_CI95_lower=theta_RE - zcrit * se_RE,
                RD_CI95_upper=theta_RE + zcrit * se_RE,
                m_runs=k,
                method_used="DL_RE",
            )
        )

    out = (
        df.groupby(list(group_cols), as_index=False).apply(_agg).reset_index(drop=True)
    )
    return out


def combine_random_effects_HKSJ(
    df: pd.DataFrame,
    group_cols=("method", "outcome"),
    ci: float = 0.95,
) -> pd.DataFrame:
    """
    Random-effects meta-analysis of RD with Hartung–Knapp–Sidik–Jonkman (HKSJ) adjustment.

    Expects per-run risk differences (RD) and their within-run variances (SE_RD^2).
    For each group (e.g., per method/outcome), it:
      1) Estimates between-run heterogeneity tau^2 using DerSimonian–Laird (DL).
      2) Pools with weights w*_r = 1 / (SE_r^2 + tau^2).
      3) Uses HKSJ small-sample adjustment for SE and t-based inference with df = m-1.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: RD, SE_RD, and the columns in `group_cols`.
    group_cols : tuple/list of str
        Columns to group by when pooling (default: ("method", "outcome")).
    ci : float
        Confidence level for intervals (default: 0.95).

    Returns
    -------
    pd.DataFrame
        One row per group with columns:
        - RD: pooled risk difference (random effects)
        - SE_RD: HKSJ-adjusted standard error
        - z: normal-approx z-statistic (NaN if undefined)
        - t: t-statistic (NaN if m < 2)
        - p_value: two-sided p-value (t with df=m-1 if m>=2; normal if m<2)
        - RD_CI95_lower / RD_CI95_upper: CI bounds at the requested level
        - tau2: between-run heterogeneity (DL)
        - I2: percent variance due to heterogeneity
        - m_runs: number of runs pooled
        - df: degrees of freedom (m-1 if m>=2 else NaN)
        - method_used: "DL_RE_HKSJ"
    """
    alpha = 1.0 - ci

    def _agg(g: pd.DataFrame) -> pd.Series:
        yi = g["RD"].astype(float).values
        vi = (g["SE_RD"].astype(float).values) ** 2
        m = len(yi)

        # If only one run: fall back to that run's estimate (z-approx)
        if m == 0:
            return pd.Series(
                dict(
                    RD=np.nan,
                    SE_RD=np.nan,
                    z=np.nan,
                    t=np.nan,
                    p_value=np.nan,
                    RD_CI95_lower=np.nan,
                    RD_CI95_upper=np.nan,
                    tau2=np.nan,
                    I2=np.nan,
                    m_runs=0,
                    df=np.nan,
                    method_used="DL_RE_HKSJ",
                )
            )
        if m == 1:
            theta = yi[0]
            se = np.sqrt(vi[0])
            z = theta / se if se > 0 else np.nan
            p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
            zcrit = norm.ppf(1 - alpha / 2)
            lo, hi = theta - zcrit * se, theta + zcrit * se
            return pd.Series(
                dict(
                    RD=theta,
                    SE_RD=se,
                    z=z,
                    t=np.nan,
                    p_value=p,
                    RD_CI95_lower=lo,
                    RD_CI95_upper=hi,
                    tau2=0.0,
                    I2=0.0,
                    m_runs=1,
                    df=np.nan,
                    method_used="DL_RE_HKSJ",
                )
            )

        # Fixed-effect quantities (for Q and DL tau^2)
        wi = 1.0 / vi
        theta_FE = np.sum(wi * yi) / np.sum(wi)
        Q = np.sum(wi * (yi - theta_FE) ** 2)
        df_Q = m - 1
        c = np.sum(wi) - (np.sum(wi**2) / np.sum(wi))
        tau2 = max((Q - df_Q) / c, 0.0) if c > 0 else 0.0

        # Random-effects weights
        w_star = 1.0 / (vi + tau2)
        theta_RE = np.sum(w_star * yi) / np.sum(w_star)

        # HKSJ variance: se_HK^2 = sum(w*_r (y_r - theta_RE)^2) / ((m-1) * sum(w*_r))
        num = np.sum(w_star * (yi - theta_RE) ** 2)
        den = (m - 1) * np.sum(w_star)
        if den > 0:
            se_HK = np.sqrt(num / den)
        else:
            # fallback to classic RE SE if degenerate
            se_HK = np.sqrt(1.0 / np.sum(w_star))

        # t-based inference with df = m-1
        df_hk = m - 1
        tcrit = tdist.ppf(1 - alpha / 2, df=df_hk)
        z = theta_RE / se_HK if se_HK > 0 else np.nan
        tstat = z
        p = 2 * (1 - tdist.cdf(abs(tstat), df=df_hk)) if np.isfinite(tstat) else np.nan

        lo, hi = theta_RE - tcrit * se_HK, theta_RE + tcrit * se_HK

        # Heterogeneity I^2 = max(0, (Q - df)/Q)
        I2 = max(0.0, (Q - df_Q) / Q) * 100.0 if Q > 0 else 0.0

        return pd.Series(
            dict(
                RD=theta_RE,
                SE_RD=se_HK,
                z=z,
                t=tstat,
                p_value=p,
                RD_CI95_lower=lo,
                RD_CI95_upper=hi,
                tau2=tau2,
                I2=I2,
                m_runs=m,
                df=df_hk,
                method_used="DL_RE_HKSJ",
            )
        )

    out = (
        df.groupby(list(group_cols), as_index=False).apply(_agg).reset_index(drop=True)
    )
    return out
