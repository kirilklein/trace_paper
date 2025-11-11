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

def pool_arm_logits(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    eta_col: str,
    se_col: str,
    out_prefix: str,
    pooling: Literal[
        "fixed_effect", "random_effects_hksj", "correlation_adjusted", "simple_mean"
    ] = "fixed_effect",
    *,
    rho: Optional[float] = None,
    weight_col: Optional[str] = None,
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

        if pooling == "correlation_adjusted":
            raw_w = None
            if weight_col is not None and (weight_col in cleaned.columns):
                raw_w = cleaned[weight_col].astype(float).values
            res_theta, res_se, res_df, _, _, _ = correlation_adjusted_arm_pool(
                yi, vi, weights=raw_w, rho=rho
            )
            return pd.Series(
                {
                    out_prefix: res_theta,
                    f"{out_prefix}_se": res_se,
                    f"{out_prefix}_tau2": np.nan,
                    "n_runs_used": m,
                    "df": res_df,
                    "method_used": "correlation_adjusted",
                }
            )

        if pooling == "simple_mean":
            theta = float(np.mean(yi)) if m > 0 else np.nan
            if m >= 2:
                s = float(np.std(yi, ddof=1))
                se_hat = s / np.sqrt(m)
                df_s = m - 1
            else:
                se_hat = np.nan
                df_s = np.nan
            return pd.Series(
                {
                    out_prefix: theta,
                    f"{out_prefix}_se": se_hat,
                    f"{out_prefix}_tau2": np.nan,
                    "n_runs_used": m,
                    "df": df_s,
                    "method_used": "simple_mean",
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
    """
    Compute inference on the pooled logit difference (η1 - η0) and
    make that the primary p-value/z reported downstream.

    Output columns preserved exactly:
      - eta_diff, se_eta_diff, df_logit, t_logit, p_value_logit,
        eta_diff_CI95_lower, eta_diff_CI95_upper,
        and the promoted p_value/z (with old ones saved as *_delta).
    """
    if df.empty:
        return df

    alpha = 1.0 - ci
    out = df.copy()

    # Clearer internal names (columns unchanged)
    treat_logit   = out["eta1_pooled"].astype(float).to_numpy()
    control_logit = out["eta0_pooled"].astype(float).to_numpy()
    treat_se      = out["eta1_pooled_se"].astype(float).to_numpy()
    control_se    = out["eta0_pooled_se"].astype(float).to_numpy()

    # Difference and its SE on the logit scale
    diff_logit = treat_logit - control_logit
    diff_se    = np.sqrt(treat_se**2 + control_se**2)

    # t-statistic for the logit difference
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.divide(diff_logit, diff_se,
                           out=np.full_like(diff_logit, np.nan),
                           where=diff_se > 0)

    # Degrees of freedom: keep existing behavior
    df1 = out.get("eta1_df", pd.Series(np.nan, index=out.index)).astype(float).to_numpy()
    if "eta0_df" in out.columns:
        df0 = out["eta0_df"].astype(float).to_numpy()
        df_logit = np.fmin(df1, df0)  # original: min(df1, df0)
    else:
        df_logit = df1                 # other file version: use eta1_df only

    # Allocate arrays
    p_vals = np.full_like(t_stat, np.nan)
    tcrit  = np.full_like(t_stat, np.nan)

    # Where we have finite df >= 1 → t distribution
    t_mask = np.isfinite(df_logit) & (df_logit >= 1) & np.isfinite(t_stat)
    if t_mask.any():
        p_vals[t_mask] = 2.0 * (1.0 - tdist.cdf(np.abs(t_stat[t_mask]), df=df_logit[t_mask]))
        tcrit[t_mask]  = tdist.ppf(1 - alpha / 2, df=df_logit[t_mask])

    # Else → normal approximation
    z_mask = (~t_mask) & np.isfinite(t_stat)
    if z_mask.any():
        p_vals[z_mask] = 2.0 * (1.0 - norm.cdf(np.abs(t_stat[z_mask])))
        tcrit[z_mask]  = norm.ppf(1 - alpha / 2)

    # If SE is invalid/nonpositive, CI is undefined
    bad_se_mask = ~np.isfinite(diff_se) | (diff_se <= 0)
    if bad_se_mask.any():
        tcrit[bad_se_mask] = np.nan

    ci_lo = diff_logit - tcrit * diff_se
    ci_hi = diff_logit + tcrit * diff_se

    # Write outputs with the same column names as before
    out["eta_diff"]             = diff_logit
    out["se_eta_diff"]          = diff_se
    out["df_logit"]             = df_logit
    out["t_logit"]              = t_stat
    out["p_value_logit"]        = p_vals
    out["eta_diff_CI95_lower"]  = ci_lo
    out["eta_diff_CI95_upper"]  = ci_hi

    # Promote logit-scale inference to primary p-value / z, preserve delta-method
    out.rename(columns={"p_value": "p_value_delta", "z": "z_delta"}, inplace=True)
    out["p_value"] = out["p_value_logit"]
    out["z"]       = out["t_logit"]
    return out


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
# Orchestrator
# -----------------------------
def compute_rd_pvalues(
    df: pd.DataFrame,
    group_cols: Optional[Union[str, List[str]]] = None,
    pooling_method: str = "inverse_variance_arms",
    arm_pooling: Literal["fixed_effect", "random_effects_hksj"] = "random_effects_hksj",
    arm_pooling_rho: Optional[float] = None,
    arm_weight_col: Optional[str] = None,
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
      - ``\"correlation_adjusted\"``: correlation-aware weighted pooling with B_eff
      - ``\"simple_mean\"``: unweighted mean of logits; SEM from sample std (ddof=1)/sqrt(m)

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
        rho=arm_pooling_rho,
        weight_col=arm_weight_col,
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
        rho=arm_pooling_rho,
        weight_col=arm_weight_col,
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
