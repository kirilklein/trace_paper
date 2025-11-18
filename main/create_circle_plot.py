"""Create circular (polar) plots for log risk ratios by ATC group.

This script:
- pools arm-level estimates across runs to obtain pooled arm probabilities
  (p1_hat, p0_hat), log risk ratios (log_RR), and p-values
- adjusts p-values for multiple testing (q-values)
- derives an ATC group (first letter of the outcome code)
- produces a circular plot with:
    * bars arranged circularly, one per outcome
    * bar height: magnitude of log RR
    * bar color: direction (positive/negative) and significance level
    * ATC group labels positioned at group centers
    * custom legend showing significance levels
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from main.helpers import ensure_output_directory, print_dataset_overview
from trace.constants import METHODS_WITH_ARMS
from trace.io import (
    filter_methods_with_arm_cis,
    load_estimates,
)
from trace.plotting.circle import plot_circle
from trace.plotting.volcano import adjust_pvalues
from trace.statistics import compute_rd_pvalues


# -----------------------------------------------------------------------------
# CLI and main orchestration
# -----------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create circular (polar) plots for log-RR by ATC group",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/semaglutide"),
        help="Directory containing input data files (combined_estimates.txt)",
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
            "correlation_adjusted",
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
        "--exclude-outcomes",
        type=str,
        default="",
        help="Comma-separated list of outcome codes to exclude (e.g., 'A10BJ,A10BK')",
    )
    parser.add_argument(
        "--exclude-groups",
        type=str,
        default="",
        help="Comma-separated list of ATC groups (first letter) to exclude (e.g., 'V,W')",
    )
    parser.add_argument(
        "--min-prevalence",
        type=float,
        default=0.01,
        help="Minimum prevalence threshold (applies to min of both arms)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_root: Path = args.output_dir
    method: str = args.method

    # Construct output directory similar to other plotting scripts
    input_folder_name = input_dir.name
    output_dir = output_root / input_folder_name / args.adjust / args.arm_pooling

    estimates_path = input_dir / "combined_estimates.txt"

    print(f"Method: {method}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Adjustment: {args.adjust} (scope={args.adjust_per})")
    print(f"Arm pooling: {args.arm_pooling}")
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
    # Apply prevalence filter
    # ------------------------------------------------------------------
    print(f"\nApplying prevalence filter (min threshold: {args.min_prevalence:.3f})...")
    n_before_prev = len(df_pooled)
    min_prev = df_pooled[["p0_hat", "p1_hat"]].min(axis=1)
    df_pooled = df_pooled[min_prev >= args.min_prevalence].copy()
    n_after_prev = len(df_pooled)
    
    if n_before_prev != n_after_prev:
        print(
            f"Filtered out {n_before_prev - n_after_prev} outcomes below prevalence threshold "
            f"({n_after_prev} remaining)"
        )
    
    if df_pooled.empty:
        print("No outcomes remaining after prevalence filter. Exiting.")
        return
    
    # Show prevalence range of remaining outcomes
    remaining_min_prev = df_pooled[["p0_hat", "p1_hat"]].min(axis=1)
    print(
        f"Prevalence range (min of arms): "
        f"min={remaining_min_prev.min():.4f}, "
        f"median={remaining_min_prev.median():.4f}, "
        f"max={remaining_min_prev.max():.4f}"
    )

    # ------------------------------------------------------------------
    # Prepare plotting dataframe
    # ------------------------------------------------------------------
    df_plot = df_pooled[["method", "outcome", "log_RR", "q_value"]].copy()

    # Drop rows without finite log_RR
    df_plot = df_plot.replace([np.inf, -np.inf], np.nan)
    n_before = len(df_plot)
    df_plot = df_plot.dropna(subset=["log_RR"]).copy()
    n_after = len(df_plot)
    if n_before != n_after:
        print(
            f"\nDropped {n_before - n_after} rows with non-finite log_RR "
            f"({n_after} remaining)"
        )

    if df_plot.empty:
        print("No rows with finite log_RR. Exiting.")
        return

    # Derive ATC group (first letter of outcome)
    df_plot["group"] = df_plot["outcome"].astype(str).str[0]

    # Apply optional filters for outcomes and groups
    n_before_filter = len(df_plot)
    if args.exclude_outcomes:
        exclude_list = [
            o.strip() for o in args.exclude_outcomes.split(",") if o.strip()
        ]
        if exclude_list:
            df_plot = df_plot[~df_plot["outcome"].isin(exclude_list)].copy()
            print(
                f"\nExcluded {len(exclude_list)} outcome(s): {', '.join(exclude_list)}"
            )

    if args.exclude_groups:
        exclude_list = [g.strip() for g in args.exclude_groups.split(",") if g.strip()]
        if exclude_list:
            df_plot = df_plot[~df_plot["group"].isin(exclude_list)].copy()
            print(f"Excluded {len(exclude_list)} group(s): {', '.join(exclude_list)}")

    n_after_filter = len(df_plot)
    if n_before_filter != n_after_filter:
        print(
            f"Filtered out {n_before_filter - n_after_filter} outcomes ({n_after_filter} remaining)"
        )

    if df_plot.empty:
        print("No rows remaining after filtering. Exiting.")
        return

    # Sort by outcome for consistent ordering
    df_plot = df_plot.sort_values("outcome").reset_index(drop=True)

    print(f"\nPrepared {len(df_plot)} outcomes for plotting")
    group_counts = df_plot["group"].value_counts().sort_index()
    print(f"ATC groups: {', '.join(f'{g}({n})' for g, n in group_counts.items())}")

    # Count significant outcomes
    n_sig_05 = (df_plot["q_value"] < 0.05).sum()
    n_sig_01 = (df_plot["q_value"] < 0.01).sum()
    n_sig_001 = (df_plot["q_value"] < 0.001).sum()
    print(
        f"Significant outcomes: {n_sig_05} (q<0.05), {n_sig_01} (q<0.01), "
        f"{n_sig_001} (q<0.001)"
    )

    # ------------------------------------------------------------------
    # Build circular plot
    # ------------------------------------------------------------------
    print("\nCreating circular plot...")
    try:
        fig = plot_circle(
            df_plot,
            outcome_col="outcome",
            log_rr_col="log_RR",
            q_value_col="q_value",
            group_col="group",
        )

        ensure_output_directory(output_dir)
        output_png = output_dir / f"circle_plot_log_rr_{method}.png"
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved circular plot to: {output_png}")
    except ValueError as e:
        print(f"Error creating plot: {e}")
        return

    print("\nDone.")


if __name__ == "__main__":
    main()
