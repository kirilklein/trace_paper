"""
Volcano Plot Module

This module provides functions for creating volcano plots to visualize
risk differences and their statistical significance across multiple outcomes.
Volcano plots show effect sizes (x-axis) versus significance (y-axis),
making it easy to identify outcomes with large and statistically significant
effects.

Key Features:
- Multiple p-value adjustment methods (Benjamini-Hochberg, Bonferroni, none)
- Per-method or global adjustment strategies
- Customizable significance thresholds and colors
- Automatic annotation of top hits
- Support for multiple methods in separate panels

Typical workflow:
1. Start with DataFrame containing RD and p-values
2. Prepare data with adjust_pvalues or prepare_volcano_data
3. Create plots with volcano_plot_per_method
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Iterable, Literal, Tuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes


# ------------------------------------------------------------------------------------
# P-VALUE ADJUSTMENT (configurable, default = Benjamini–Hochberg / FDR)
# ------------------------------------------------------------------------------------
def adjust_pvalues(
    pvals: Iterable[float],
    method: Literal["bh", "bonferroni", "none"] = "bh",
) -> np.ndarray:
    """
    Return adjusted p-values (q-values) in the original order.

    Parameters
    ----------
    pvals : iterable of float
        Array of p-values to adjust
    method : {'bh', 'bonferroni', 'none'}, optional
        Adjustment method:
        - 'bh': Benjamini-Hochberg step-up FDR control (default)
        - 'bonferroni': Bonferroni family-wise error rate control
        - 'none': No adjustment (q = p)

    Returns
    -------
    np.ndarray
        Adjusted p-values (q-values) in original order

    Notes
    -----
    The Benjamini-Hochberg procedure controls the False Discovery Rate (FDR),
    which is less conservative than family-wise error rate control methods
    like Bonferroni. Non-finite p-values (NaN, inf) are preserved as NaN
    in the output.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size

    if method == "none":
        return p.copy()

    if method == "bonferroni":
        return np.minimum(p * n, 1.0)

    if method == "bh":
        # Benjamini–Hochberg (step-up) on finite p-values
        finite_mask = np.isfinite(p)
        q = np.full_like(p, np.nan, dtype=float)
        if finite_mask.sum() == 0:
            return q  # all NaN

        ps = p[finite_mask]
        order = np.argsort(ps)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, ps.size + 1)

        # Compute BH: q_i = (m/i) * p_i
        q_raw = ps.size / ranks * ps
        # Make it monotone non-increasing backward
        q_monotone = np.minimum.accumulate(q_raw[order[::-1]])[::-1]
        q_vals = np.minimum(q_monotone, 1.0)
        q[finite_mask] = q_vals[ranks - 1]  # place back in original positions
        return q

    raise ValueError(f"Unknown method: {method}")


# ------------------------------------------------------------------------------------
# DATA PREP
# ------------------------------------------------------------------------------------
def prepare_volcano_data(
    df: pd.DataFrame,
    rd_col: str = "RD",
    p_col: str = "p_value",
    method_col: str = "method",
    outcome_col: str = "outcome",
    adjust: Literal["bh", "bonferroni", "none"] = "bh",
    adjust_per: Literal["by_method", "global"] = "by_method",
    p_floor: float = 1e-300,
) -> pd.DataFrame:
    """
    Prepare data for volcano plotting.

    Computes adjusted p-values and -log10(p) transformation needed for
    volcano plots. Can adjust p-values either within each method separately
    or globally across all methods.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with RD estimates and p-values
    rd_col : str, optional
        Column name for risk difference (default: "RD")
    p_col : str, optional
        Column name for p-values (default: "p_value")
    method_col : str, optional
        Column name for method identifier (default: "method")
    outcome_col : str, optional
        Column name for outcome identifier (default: "outcome")
    adjust : {'bh', 'bonferroni', 'none'}, optional
        P-value adjustment method (default: "bh")
    adjust_per : {'by_method', 'global'}, optional
        Whether to adjust within each method or globally (default: "by_method")
    p_floor : float, optional
        Minimum p-value for -log10 transformation to avoid inf (default: 1e-300)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: method, outcome, RD, p_value, q_value, neglog10p
    """
    d = df.copy()

    if adjust_per == "by_method":
        q = np.empty(len(d), dtype=float)
        for m, idx in d.groupby(method_col).groups.items():
            q[idx] = adjust_pvalues(d.loc[idx, p_col].values, method=adjust)
        d["q_value"] = q
    else:  # "global"
        d["q_value"] = adjust_pvalues(d[p_col].values, method=adjust)

    # Robust -log10 p
    d["neglog10p"] = -np.log10(np.clip(d[p_col].values, p_floor, 1.0))

    return d[[method_col, outcome_col, rd_col, p_col, "q_value", "neglog10p"]].rename(
        columns={rd_col: "RD", p_col: "p_value"}
    )


# ------------------------------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------------------------------
def volcano_plot_per_method(
    df_volcano: pd.DataFrame,
    alpha: float = 0.05,
    method_col: str = "method",
    outcome_col: str = "outcome",
    label_map: Optional[Dict[str, str]] = None,
    max_labels_per_panel: int = 10,
    figsize_per_panel: Tuple[float, float] = (6, 4),
    point_size: int = 14,
    sig_color: Optional[str] = None,
    ns_color: Optional[str] = None,
) -> Tuple[Figure, list]:
    """
    Create volcano plots with one panel per method.

    Each panel shows risk difference (x-axis) vs -log10(p-value) (y-axis),
    with points colored by significance (q < alpha). Includes reference lines
    at the significance threshold and RD = 0, and annotates top hits.

    Parameters
    ----------
    df_volcano : pd.DataFrame
        Prepared volcano data from prepare_volcano_data with columns:
        method, outcome, RD, p_value, q_value, neglog10p
    alpha : float, optional
        Significance threshold for q-values (default: 0.05)
    method_col : str, optional
        Column name for method identifier (default: "method")
    outcome_col : str, optional
        Column name for outcome identifier (default: "outcome")
    label_map : dict, optional
        Dictionary mapping outcome names to display labels (default: None)
    max_labels_per_panel : int, optional
        Maximum number of outcomes to annotate per panel (default: 10)
    figsize_per_panel : tuple of (width, height), optional
        Figure size per panel in inches (default: (6, 4))
    point_size : int, optional
        Marker size for scatter points (default: 14)
    sig_color : str, optional
        Color for significant points (default: None, uses matplotlib default)
    ns_color : str, optional
        Color for non-significant points (default: None, uses matplotlib default)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing all panels
    axes : list of matplotlib.axes.Axes
        List of axes objects, one per method

    Notes
    -----
    - Horizontal line shows -log10(alpha) significance threshold
    - Vertical line at RD = 0 shows null hypothesis
    - Top hits are labeled based on smallest p-values
    """
    methods = list(df_volcano[method_col].unique())
    n = len(methods)
    if n == 0:
        raise ValueError("No methods found in dataframe.")

    # Figure sizing
    w, h = figsize_per_panel
    fig, axes = plt.subplots(1, n, figsize=(w * n, h), sharey=True)
    if n == 1:
        axes = [axes]

    y_thr = -np.log10(alpha)

    for ax, m in zip(axes, methods):
        d = df_volcano[df_volcano[method_col] == m].copy()
        if d.empty:
            ax.set_title(str(m))
            ax.set_xlabel("Risk difference (RD)")
            ax.set_ylabel("-log10(p-value)")
            continue

        # Significant mask by q < alpha
        sig = d["q_value"] < alpha

        # Colors (allow user overrides, else use cycle)
        c_sig = sig_color
        c_ns = ns_color

        # Non-significant points
        ax.scatter(
            d.loc[~sig, "RD"],
            d.loc[~sig, "neglog10p"],
            s=point_size,
            alpha=0.7,
            label="q ≥ α",
            color=c_ns,
        )
        # Significant points
        ax.scatter(
            d.loc[sig, "RD"],
            d.loc[sig, "neglog10p"],
            s=point_size,
            alpha=0.9,
            label="q < α",
            color=c_sig,
        )

        # Guide lines
        ax.axhline(y_thr, linestyle="--", linewidth=1, color="gray", alpha=0.5)
        ax.axvline(0.0, linestyle="--", linewidth=1, color="gray", alpha=0.5)

        # Labels: top hits by -log10 p (limit per panel)
        top = d.sort_values("neglog10p", ascending=False).head(max_labels_per_panel)
        for _, r in top.iterrows():
            name = str(r[outcome_col])
            if label_map and name in label_map:
                name = label_map[name]
            ax.annotate(
                name,
                (r["RD"], r["neglog10p"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=9,
            )

        ax.set_title(str(m))
        ax.set_xlabel("Risk difference (RD)")
        ax.grid(alpha=0.2, linestyle=":", linewidth=0.8)

    axes[0].set_ylabel("-log10(p-value)")
    handles, labels = axes[-1].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    return fig, axes
