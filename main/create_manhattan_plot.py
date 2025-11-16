"""Create prevalence vs log-RR Manhattan-style plots for a single method.

This script:
- pools arm-level estimates across runs to obtain pooled arm probabilities
  (p1_hat, p0_hat), log risk ratios (log_RR), and p-values
- adjusts p-values for multiple testing (q-values)
- merges in mean prevalence per outcome from ``combined_stats.txt``
- derives an ATC group (first letter of the outcome code)
- produces a faceted scatter plot with:
    * x-axis: treated (or total) prevalence
    * y-axis: log relative risk (log_RR)
    * one panel per ATC group
    * point colors encoding sign (positive/negative) and significance level
"""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main.helpers import ensure_output_directory, print_dataset_overview
from trace.constants import METHODS_WITH_ARMS
from trace.io import (
    PrevalenceStats,
    filter_methods_with_arm_cis,
    load_estimates,
    load_prevalence_statistics,
    rename_prevalence_columns,
)
from trace.plotting.volcano import adjust_pvalues
from trace.statistics import compute_rd_pvalues


# -----------------------------------------------------------------------------
# CLI and main orchestration
# -----------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create prevalence vs log-RR Manhattan-style plots for a single method",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/semaglutide"),
        help="Directory containing input data files (combined_estimates.txt, combined_stats.txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Base directory for output figures",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="IPW",
        help="Estimation method to plot (must have arm-level CIs)",
    )
    parser.add_argument(
        "--adjust",
        choices=[
            "bh",
            "by",
            "tsbh",
            "tsbky",
            "bonferroni",
            "sidak",
            "holm",
            "holm-sidak",
            "hochberg",
            "hommel",
            "none",
        ],
        default="bh",
        help="Multiple testing adjustment method",
    )
    parser.add_argument(
        "--adjust-per",
        dest="adjust_per",
        choices=["by_method", "global"],
        default="by_method",
        help="Scope of multiple testing adjustment (only one method is plotted here)",
    )
    parser.add_argument(
        "--arm-pooling",
        choices=[
            "random_effects_hksj",
            "simple_mean",
            "rubins_rules",
            "inter_intra_variance",
        ],
        default="simple_mean",
        help=(
            "Arm-level pooling on the logit scale across runs. "
            "See trace.statistics.pool_arm_logits for details."
        ),
    )
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fast mode (reserved for future extensions; currently always produces the main plot)",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Verbose output",
    )
    parser.add_argument(
        "--annotate-top",
        type=int,
        default=5,
        help="Number of top significant codes to annotate per ATC group panel (0 to disable)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_root: Path = args.output_dir
    method: str = args.method

    # Construct output directory similar to create_volcano_plot.py
    input_folder_name = input_dir.name
    output_dir = output_root / input_folder_name / args.adjust / args.arm_pooling

    estimates_path = input_dir / "combined_estimates.txt"
    stats_path = input_dir / "combined_stats.txt"

    print(f"Method: {method}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Adjustment: {args.adjust} (scope={args.adjust_per})")
    print(f"Arm pooling: {args.arm_pooling}")
    print(f"Annotate top: {args.annotate_top} codes per panel")
    if args.fast:
        print("Fast mode: enabled (currently identical to default mode)")
    print()

    # ------------------------------------------------------------------
    # Load arm-level estimates and restrict to methods with arm CIs
    # ------------------------------------------------------------------
    df_raw = load_estimates(estimates_path)
    print_dataset_overview(df_raw)

    df_with_arms = filter_methods_with_arm_cis(df_raw, METHODS_WITH_ARMS)
    print(
        f"\nFiltered to methods with arm-level CIs {METHODS_WITH_ARMS}: {len(df_with_arms)} rows"
    )

    if method not in df_with_arms["method"].unique():
        raise ValueError(
            f"Requested method '{method}' not present in filtered estimates. "
            f"Available methods with arms: {sorted(df_with_arms['method'].unique())}"
        )

    df_method = df_with_arms[df_with_arms["method"] == method].copy()

    # ------------------------------------------------------------------
    # Pool arms across runs to obtain p-values and arm probabilities
    # ------------------------------------------------------------------
    print("\nPooling across runs using arm-level pooling on the logit scale...")
    df_pooled = compute_rd_pvalues(
        df_method,
        group_cols=("method", "outcome"),
        arm_pooling=args.arm_pooling,  # type: ignore[arg-type]
        verbose=args.verbose,
    )
    print(f"Computed {len(df_pooled)} pooled {method} method-outcome combinations")

    if df_pooled.empty:
        print("No pooled results available for the requested method. Exiting.")
        return

    required_cols = {"p1_hat", "p0_hat", "p_value"}
    if not required_cols.issubset(df_pooled.columns):
        missing = required_cols - set(df_pooled.columns)
        raise ValueError(
            "Pooled dataframe is missing required columns for RR/log-RR computation: "
            + ", ".join(sorted(missing))
        )

    # Derive RR and log_RR from pooled arm probabilities
    df_pooled = df_pooled.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df_pooled["RR"] = df_pooled["p1_hat"] / df_pooled["p0_hat"]
        df_pooled["log_RR"] = np.log(df_pooled["RR"])

    # ------------------------------------------------------------------
    # Adjust p-values for multiple testing to obtain q-values
    # ------------------------------------------------------------------
    print("\nAdjusting p-values for multiple testing...")
    if args.adjust_per == "by_method":
        # Only a single method is present; adjust within this subset
        df_pooled["q_value"] = adjust_pvalues(
            df_pooled["p_value"].values,
            method=args.adjust,  # type: ignore[arg-type]
        )
    else:  # "global" (equivalent here since only one method is included)
        df_pooled["q_value"] = adjust_pvalues(
            df_pooled["p_value"].values,
            method=args.adjust,  # type: ignore[arg-type]
        )

    # Basic diagnostics on effects and p-values
    finite_rr = df_pooled["RR"].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite_rr.empty:
        print(
            f"RR summary: min={finite_rr.min():.3f}, "
            f"median={finite_rr.median():.3f}, max={finite_rr.max():.3f}"
        )
    finite_log_rr = df_pooled["log_RR"].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite_log_rr.empty:
        print(
            f"log_RR summary: min={finite_log_rr.min():.3f}, "
            f"median={finite_log_rr.median():.3f}, max={finite_log_rr.max():.3f}"
        )

    finite_p = df_pooled["p_value"].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite_p.empty:
        print(
            f"p-value summary: min={finite_p.min():.3e}, "
            f"median={finite_p.median():.3e}, max={finite_p.max():.3e}"
        )

    finite_q = df_pooled["q_value"].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite_q.empty:
        print(
            f"q-value summary: min={finite_q.min():.3e}, "
            f"median={finite_q.median():.3e}, max={finite_q.max():.3e}"
        )

    # ------------------------------------------------------------------
    # Load prevalence statistics and merge with pooled results
    # ------------------------------------------------------------------
    print("\nLoading prevalence statistics...")
    prevalence_stats: PrevalenceStats = load_prevalence_statistics(stats_path)
    prevalence_summary = rename_prevalence_columns(prevalence_stats.summary)

    # Prefer treated mean prevalence as x-axis; fall back to total if needed
    prevalence_col_candidates: List[str] = [
        "prevalence_mean_treated",
        "prevalence_mean_total",
    ]
    available_prevalence_cols = [
        c for c in prevalence_col_candidates if c in prevalence_summary.columns
    ]
    if not available_prevalence_cols:
        raise ValueError(
            "No suitable prevalence mean column found in prevalence summary. "
            f"Checked: {', '.join(prevalence_col_candidates)}"
        )
    prevalence_col = available_prevalence_cols[0]
    print(f"Using prevalence column '{prevalence_col}' for x-axis")

    df_plot = df_pooled[["method", "outcome", "log_RR", "q_value"]].merge(
        prevalence_summary[["outcome", prevalence_col]], on="outcome", how="left"
    )
    df_plot = df_plot.rename(columns={prevalence_col: "prevalence"})

    # Drop rows without prevalence or finite log_RR
    df_plot = df_plot.replace([np.inf, -np.inf], np.nan)
    n_before = len(df_plot)
    df_plot = df_plot.dropna(subset=["prevalence", "log_RR"]).copy()
    n_after = len(df_plot)
    print(
        f"Merged pooled results with prevalence: {n_after} rows "
        f"(dropped {n_before - n_after} with missing data)"
    )

    # Prevalence diagnostics
    if not df_plot.empty:
        print(
            "Prevalence range: "
            f"min={df_plot['prevalence'].min():.4f}, "
            f"median={df_plot['prevalence'].median():.4f}, "
            f"max={df_plot['prevalence'].max():.4f}"
        )

    if df_plot.empty:
        print("No rows with both log_RR and prevalence. Exiting.")
        return

    # ------------------------------------------------------------------
    # Derive ATC group (first letter of outcome) and prepare plotting dataframe
    # ------------------------------------------------------------------
    df_plot["group"] = df_plot["outcome"].astype(str).str[0]
    group_order = sorted(df_plot["group"].dropna().unique())

    # ------------------------------------------------------------------
    # Significance binning and color mapping (inspired by plot_radar.ipynb)
    # ------------------------------------------------------------------
    print("\nComputing significance bins and colors...")
    # Thresholds on q-value: standard significance levels
    thresholds = np.array([0.05, 0.01, 0.001])
    bins = np.digitize(df_plot["q_value"].to_numpy(), thresholds)

    # Color palettes (light to dark)
    reds = np.array(["#ffb3b3", "#bf3a3a", "#9c0202"], dtype=object)
    blues = np.array(["#b3c6ff", "#4d79ff", "#0033cc"], dtype=object)
    neutral = "lightgrey"

    colors: List[str] = []
    for log_rr, q, b in zip(
        df_plot["log_RR"].to_numpy(), df_plot["q_value"].to_numpy(), bins
    ):
        if np.isnan(q) or q > 0.05:
            colors.append(neutral)
            continue
        idx = min(b - 1, len(reds) - 1)
        if log_rr > 0:
            colors.append(reds[idx])
        else:
            colors.append(blues[idx])

    df_plot["color"] = colors

    # ------------------------------------------------------------------
    # Build Manhattan-style faceted plot (prevalence vs log_RR per ATC group)
    # ------------------------------------------------------------------
    print("\nCreating prevalence vs log-RR Manhattan-style plot...")
    n_groups = len(group_order)
    if n_groups == 0:
        print("No ATC groups present after filtering. Exiting.")
        return

    n_cols = min(4, n_groups)
    n_rows = int(ceil(n_groups / n_cols))

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        sharex=True,
        sharey=True,
    )

    # Flatten axes array for easy indexing
    if isinstance(axs, np.ndarray):
        axs_flat: Iterable[plt.Axes] = axs.flatten()
    else:
        axs_flat = [axs]

    for i, group in enumerate(group_order):
        ax = list(axs_flat)[i]
        d = df_plot[df_plot["group"] == group]
        x = d["prevalence"].to_numpy()
        y = d["log_RR"].to_numpy()
        c = d["color"].to_numpy()

        ax.scatter(x, y, s=30, c=c)
        ax.set_title(f"ATC Group {group}", fontsize=12, y=0.9)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.grid(axis="x", linestyle="--", alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlim(0.0001, 1.0)

        # Annotate top N significant codes per panel
        if args.annotate_top > 0 and len(d) > 0:
            # Sort by q-value (most significant first) and take top N
            d_sorted = d.sort_values("q_value").head(args.annotate_top)
            for idx, row in d_sorted.iterrows():
                # Simple offset to reduce overlap
                offset_x = 1.2  # Slight offset in log space
                offset_y = 0.1 if row["log_RR"] > 0 else -0.1
                ax.annotate(
                    row["outcome"],
                    xy=(row["prevalence"], row["log_RR"]),
                    xytext=(row["prevalence"] * offset_x, row["log_RR"] + offset_y),
                    fontsize=7,
                    alpha=0.8,
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="arc3,rad=0.2",
                        alpha=0.5,
                        lw=0.5,
                    ),
                )

    # Hide unused axes (if any)
    axs_list = list(axs_flat)
    for j in range(n_groups, len(axs_list)):
        axs_list[j].set_visible(False)

    # Shared y-label on first column
    for r in range(n_rows):
        idx = r * n_cols
        if idx < n_groups:
            axs_list[idx].set_ylabel("Log Relative Risk", fontsize=11)

    # Shared x-label on bottom row
    bottom_indices: List[int] = list(range((n_rows - 1) * n_cols, n_rows * n_cols))
    for idx in bottom_indices:
        if idx >= n_groups:
            continue
        ax = axs_list[idx]
        ax.set_xlabel("Treated Prevalence", fontsize=11)
        ax.set_xticks([0.0001, 0.001, 0.01, 0.1, 1.0])

    fig.tight_layout(h_pad=0.3, w_pad=0.3)

    ensure_output_directory(output_dir)
    output_png = output_dir / f"manhattan_prevalence_log_rr_{method}.png"
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Manhattan-style plot to: {output_png}")
    print("\nDone.")


if __name__ == "__main__":
    main()
