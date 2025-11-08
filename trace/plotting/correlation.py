"""
Correlation plotting utilities for comparing effects across methods.

Provides a scatter plot comparing effects (e.g., RD or RR) between two methods
such as IPW and TMLE, along with correlation statistics.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import pearsonr, spearmanr


def plot_method_correlation(
    df: pd.DataFrame,
    *,
    methods: Sequence[str] = ("IPW", "TMLE"),
    method_col: str = "method",
    outcome_col: str = "outcome",
    effect_col: str = "RD",
    effect_label: Optional[str] = None,
    xscale: Optional[Literal["linear", "log"]] = None,
    transform: Optional[Literal["log10", "log"]] = None,
    clip_quantiles: Optional[Tuple[float, float]] = (0.005, 0.995),
    hexbin: bool = True,
    gridsize: int = 40,
    cmap: str = "viridis",
    alpha: float = 0.85,
    significance_col: str = "q_value",
    alpha_threshold: float = 0.05,
    colors: Optional[Dict[str, str]] = None,
    show_legend: bool = True,
    figsize: Tuple[float, float] = (6.5, 6.0),
    point_size: float = 30.0,
) -> Tuple[Figure, Axes]:
    """
    Plot correlation of effects between two methods (default: IPW vs TMLE).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least columns [method, outcome, <effect_col>].
    methods : Sequence[str], optional
        Two methods to compare, in (x, y) order for the plot.
    method_col : str, optional
        Column name indicating method labels.
    outcome_col : str, optional
        Column name indicating outcomes to align across methods.
    effect_col : str, optional
        Column containing the effect measure to compare (e.g., "RD" or "RR").
    effect_label : str, optional
        Label for the axes. Defaults to effect_col if not provided.
    xscale : {"linear","log"}, optional
        Scale for both axes (applied symmetrically). Defaults to None (linear).
    figsize : tuple, optional
        Figure size (width, height) in inches.
    point_size : float, optional
        Marker size for the scatter points.

    Returns
    -------
    (fig, ax) : Tuple[Figure, Axes]
        The created Matplotlib figure and axis.
    """
    if len(methods) != 2:
        raise ValueError("plot_method_correlation expects exactly two methods.")

    method_x, method_y = methods

    subset = df[[outcome_col, method_col, effect_col]].dropna(subset=[effect_col]).copy()
    subset = subset[subset[method_col].isin([method_x, method_y])]

    if subset.empty:
        raise ValueError("No data available for the requested methods.")

    pivot = subset.pivot_table(
        index=outcome_col, columns=method_col, values=effect_col, aggfunc="first"
    )

    missing_cols = [m for m in [method_x, method_y] if m not in pivot.columns]
    if missing_cols:
        raise ValueError(
            "Missing methods in the pivoted data: " + ", ".join(missing_cols)
        )

    pivot = pivot.dropna(subset=[method_x, method_y])
    if pivot.empty:
        raise ValueError("No overlapping outcomes between the selected methods.")

    x_raw = pivot[method_x].astype(float).values
    y_raw = pivot[method_y].astype(float).values

    # Optional significance-based coloring (align by outcome index)
    sign_x = None
    sign_y = None
    if significance_col in df.columns:
        sig_df = (
            df[[outcome_col, method_col, significance_col]]
            .dropna(subset=[significance_col])
            .copy()
        )
        sig_df["is_sig"] = sig_df[significance_col].astype(float) < alpha_threshold
        piv_sig = sig_df.pivot_table(
            index=outcome_col, columns=method_col, values="is_sig", aggfunc="first"
        )
        # Reindex to effect pivot index to guarantee alignment
        piv_sig = piv_sig.reindex(index=pivot.index)
        if method_x in piv_sig.columns and method_y in piv_sig.columns:
            sign_x = piv_sig[method_x].astype(bool).values
            sign_y = piv_sig[method_y].astype(bool).values

    # Optional transform (operate on data rather than axes for better visibility)
    if transform in {"log10", "log"}:
        eps = 1e-12
        x_safe = np.clip(x_raw, eps, np.inf)
        y_safe = np.clip(y_raw, eps, np.inf)
        if transform == "log10":
            x = np.log10(x_safe)
            y = np.log10(y_safe)
            label_suffix = " (log10)"
        else:
            x = np.log(x_safe)
            y = np.log(y_safe)
            label_suffix = " (log)"
    else:
        x = x_raw
        y = y_raw
        label_suffix = ""

    # Optional clipping to central quantile range (robust against extremes)
    finite_mask_xy = np.isfinite(x) & np.isfinite(y)
    if clip_quantiles is not None and finite_mask_xy.any():
        qlo, qhi = clip_quantiles
        x_lo, x_hi = np.quantile(x[finite_mask_xy], [qlo, qhi])
        y_lo, y_hi = np.quantile(y[finite_mask_xy], [qlo, qhi])
        keep = finite_mask_xy & (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
    else:
        keep = finite_mask_xy

    # Correlation statistics
    r, r_p = pearsonr(x[keep], y[keep])
    rho, rho_p = spearmanr(x[keep], y[keep])
    n = int(np.sum(keep))

    fig, ax = plt.subplots(figsize=figsize)
    if hexbin and n >= 200 and (sign_x is None or sign_y is None):
        hb = ax.hexbin(
            x[keep],
            y[keep],
            gridsize=gridsize,
            cmap=cmap,
            mincnt=1,
            linewidths=0.0,
        )
        cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Count")
    else:
        if sign_x is not None and sign_y is not None:
            # Category colors
            color_defaults = {
                "both_ns": "#000000",   # black
                "discordant": "#999999",  # grey
                "both_sig": "#d62728",  # red
            }
            if colors:
                color_defaults.update(colors)

            sx = sign_x[keep]
            sy = sign_y[keep]
            both_ns = (~sx) & (~sy)
            discordant = sx ^ sy
            both_sig = sx & sy

            # Plot each category
            if both_ns.any():
                ax.scatter(
                    x[keep][both_ns],
                    y[keep][both_ns],
                    s=point_size,
                    alpha=alpha,
                    color=color_defaults["both_ns"],
                    edgecolor="white",
                    linewidth=0.4,
                    label=f"Both non-significant (n={int(both_ns.sum())})",
                )
            if discordant.any():
                ax.scatter(
                    x[keep][discordant],
                    y[keep][discordant],
                    s=point_size,
                    alpha=alpha,
                    color=color_defaults["discordant"],
                    edgecolor="white",
                    linewidth=0.4,
                    label=f"Only one significant (n={int(discordant.sum())})",
                )
            if both_sig.any():
                ax.scatter(
                    x[keep][both_sig],
                    y[keep][both_sig],
                    s=point_size,
                    alpha=alpha,
                    color=color_defaults["both_sig"],
                    edgecolor="white",
                    linewidth=0.4,
                    label=f"Both significant (n={int(both_sig.sum())})",
                )
            if show_legend:
                ax.legend(frameon=True, fontsize=9, title=f"Significance @ α={alpha_threshold:g}")
        else:
            ax.scatter(
                x[keep],
                y[keep],
                s=point_size,
                alpha=alpha,
                color="#4C78A8",
                edgecolor="white",
                linewidth=0.6,
            )

    # Identity line
    finite_mask = keep
    if finite_mask.any():
        lim_min = np.nanmin([np.min(x[finite_mask]), np.min(y[finite_mask])])
        lim_max = np.nanmax([np.max(x[finite_mask]), np.max(y[finite_mask])])
        if np.isfinite(lim_min) and np.isfinite(lim_max):
            pad = 0.02 * (lim_max - lim_min) if np.isfinite(lim_max - lim_min) else 0.0
            lo = lim_min - pad
            hi = lim_max + pad
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="#888888", linewidth=1.0, alpha=0.7, zorder=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

    # Least-squares fit
    if finite_mask.sum() >= 2:
        slope, intercept = np.polyfit(x[finite_mask], y[finite_mask], 1)
        xs = np.array(ax.get_xlim())
        ax.plot(xs, slope * xs + intercept, color="#E45756", linewidth=1.3, alpha=0.9, zorder=2)

    xlabel = (effect_label or effect_col) + label_suffix
    ylabel = (effect_label or effect_col) + label_suffix
    ax.set_xlabel(f"{xlabel} — {method_x}")
    ax.set_ylabel(f"{ylabel} — {method_y}")

    # Apply symmetric scaling if requested
    if xscale in {"log"}:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.grid(alpha=0.2, linestyle=":", linewidth=0.8)

    # Annotation with correlation stats
    annotation = (
        f"Pearson r = {r:.3f} (p={r_p:.1e})\n"
        f"Spearman ρ = {rho:.3f} (p={rho_p:.1e})\n"
        f"n = {n}"
    )
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#dddddd", boxstyle="round,pad=0.3"),
    )

    fig.tight_layout()
    return fig, ax


