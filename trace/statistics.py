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
from scipy.stats import norm
from typing import Dict, List, Optional, Union

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
    lo: Union[float, np.ndarray], 
    hi: Union[float, np.ndarray], 
    z: float = 1.96
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
    out_prefix: str
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
    def _agg(g):
        g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=[eta_col, se_col])
        if g.empty:
            return pd.Series({
                out_prefix: np.nan,
                f"{out_prefix}_se": np.nan,
                "n_runs_used": 0
            })
        w = 1.0 / (g[se_col].values ** 2)
        eta_hat = np.sum(w * g[eta_col].values) / np.sum(w)
        se_hat = np.sqrt(1.0 / np.sum(w))
        return pd.Series({
            out_prefix: eta_hat,
            f"{out_prefix}_se": se_hat,
            "n_runs_used": int(g.shape[0])
        })

    return (
        df.groupby(group_cols, as_index=False)
          .apply(_agg, include_groups=False)
          .reset_index(drop=True)
    )


# -----------------------------
# Step C: RD inference from arm logits (delta method)
# -----------------------------
def rd_inference_from_arm_logits(
    eta1: np.ndarray,
    se1: np.ndarray,
    eta0: np.ndarray,
    se0: np.ndarray,
    z_crit: float = 1.96
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
    p1 = inv_logit(eta1)
    p0 = inv_logit(eta0)
    rd = p1 - p0

    # Delta-method variance of RD on probability scale
    var_rd = (p1 * (1 - p1))**2 * (se1**2) + (p0 * (1 - p0))**2 * (se0**2)
    se_rd = np.sqrt(var_rd)

    z = rd / se_rd
    pval = 2 * (1 - norm.cdf(np.abs(z)))
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
    group_cols: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Compute RD p-values either per row or pooled across runs.
    
    Main orchestrator function that handles the complete workflow:
    1. Add logit metrics to input data
    2. Either compute per-row RDs or pool by group then compute RDs
    3. Return DataFrame with RD estimates and inference
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with arm-level probability estimates and CIs.
        Required columns: effect_1, effect_0, effect_1_CI95_lower,
        effect_1_CI95_upper, effect_0_CI95_lower, effect_0_CI95_upper
    group_cols : str, list of str, or None, optional
        If None, compute per-row RDs. If specified, pool within each group
        (e.g., ["method", "outcome"]) before computing RDs.
    
    Returns
    -------
    pd.DataFrame
        If group_cols is None: original DataFrame with added RD columns
        If group_cols specified: one row per group with pooled RD estimates
        
        Added columns include: RD, SE_RD, z, p_value, RD_CI95_lower,
        RD_CI95_upper, p1_hat, p0_hat
    """
    dfl = add_logit_arm_metrics(df)

    if not group_cols:
        # Per-row computation
        res = rd_inference_from_arm_logits(
            eta1=dfl["eta1"].values,
            se1=dfl["se_eta1"].values,
            eta0=dfl["eta0"].values,
            se0=dfl["se_eta0"].values,
        )
        out = dfl.copy()
        for k, v in res.items():
            out[k] = v
        return out

    # Pooled over groups
    arm1 = pool_arm_logits(dfl, group_cols, "eta1", "se_eta1", "eta1_pooled")
    arm0 = pool_arm_logits(dfl, group_cols, "eta0", "se_eta0", "eta0_pooled")

    pooled = pd.merge(arm1, arm0, on=group_cols, how="inner")

    res = rd_inference_from_arm_logits(
        eta1=pooled["eta1_pooled"].values,
        se1=pooled["eta1_pooled_se"].values,
        eta0=pooled["eta0_pooled"].values,
        se0=pooled["eta0_pooled_se"].values,
    )
    for k, v in res.items():
        pooled[k] = v
    return pooled

