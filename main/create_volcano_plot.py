"""Unified script to create volcano plots for treatment effect analysis.

Supports both Risk Difference (RD) and Risk Ratio (RR) analyses with configurable
parameters via command-line interface.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from main.helpers import (
    augment_volcano_dataframe,
    ensure_output_directory,
    ensure_required_columns,
    print_dataset_overview,
    print_significance_confusion_matrix,
    summarise_per_run_effects,
)
from trace.constants import DEFAULT_ALPHA, METHODS_WITH_ARMS
from trace.diagnostics import run_diagnostics
from trace.io import (
    PrevalenceStats,
    filter_methods_with_arm_cis,
    load_atc_dictionary,
    load_estimates,
    load_prevalence_statistics,
    rename_prevalence_columns,
)
from trace.plotting.confusion import plot_confusion_matrix
from trace.plotting.volcano import (
    prepare_volcano_data,
    volcano_overlay_methods,
    volcano_plot_per_method,
)
from trace.plotting.volcano_plotly import (
    build_plotly_overlay_methods,
    build_plotly_volcano,
    save_plotly_figure,
)
from trace.statistics import compute_rd_pvalues
from trace.plotting.correlation import plot_method_correlation


# -----------------------------------------------------------------------------
# Main execution flow
# -----------------------------------------------------------------------------
def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Create volcano plots for treatment effect analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/semaglutide"),
        help="Directory containing input data files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for output figures",
    )
    parser.add_argument(
        "--effect-type",
        choices=["RD", "RR", "log-RR"],
        default="RD",
        help="Effect measure: Risk Difference (RD), Risk Ratio (RR), or log Risk Ratio",
    )
    parser.add_argument(
        "--diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run diagnostic analyses",
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
        help="Scope of multiple testing adjustment",
    )
    parser.add_argument(
        "--min-prevalence",
        type=float,
        default=0.0,
        help="Minimum prevalence threshold (as proportion 0-1). Filters outcomes with prevalence_mean_total < threshold. Default: 0.0 (no filtering)",
    )

    parser.add_argument(
        "--arm-pooling",
        choices=[
            "random_effects_hksj",
            "fixed_effect",
            "correlation_adjusted",
            "simple_mean",
            "rubins_rules",
        ],
        default="simple_mean",
        help=(
            "Arm-level pooling on the logit scale across runs: "
            "'random_effects_hksj' (DerSimonian–Laird with HKSJ SE) or "
            "'fixed_effect' (inverse-variance fixed effect) or "
            "'correlation_adjusted' (uses weights and rho) or "
            "'simple_mean' (unweighted mean of logits; SEM uses sample std with ddof=1)"
        ),
    )

    parser.add_argument(
        "--arm-pooling-rho",
        type=float,
        default=None,
        help=(
            "Correlation parameter rho for 'correlation_adjusted' arm pooling. "
            "If omitted, a default internal value is used."
        ),
    )
    parser.add_argument(
        "--arm-weight-col",
        type=str,
        default=None,
        help=(
            "Optional column name for run weights/sizes used by "
            "'correlation_adjusted' arm pooling. Defaults to equal weights."
        ),
    )

    args = parser.parse_args()

    # Construct file paths
    estimates_path = args.input_dir / "combined_estimates.txt"
    stats_path = args.input_dir / "combined_stats.txt"

    # Determine effect parameters
    effect_type = args.effect_type
    if effect_type == "log-RR":
        effect_col = "RR"
        effect_label = "Risk ratio (RR)"
        null_value = 1.0
        xscale = "log"
        effect_alias = "RR"
    elif effect_type == "RR":
        effect_col = "RR"
        effect_label = "Risk ratio (RR)"
        null_value = 1.0
        xscale = "linear"
        effect_alias = "RR"
    else:  # RD
        effect_col = "RD"
        effect_label = "Risk difference (RD)"
        null_value = 0.0
        xscale = "linear"
        effect_alias = "RD"

    # Create output suffix for files
    output_suffix = effect_type.lower().replace("-", "_")

    print(f"Effect type: {effect_type}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    diagnostics_status = "enabled" if args.diagnostics else "disabled"
    print(f"Diagnostics: {diagnostics_status}")
    print()

    # Load data
    df_raw = load_estimates(estimates_path)
    print_dataset_overview(df_raw)

    df_with_arms = filter_methods_with_arm_cis(df_raw, METHODS_WITH_ARMS)
    print(f"\nFiltered to IPW and TMLE: {len(df_with_arms)} rows")

    required_cols = [
        "effect_1",
        "effect_0",
        "effect_1_CI95_lower",
        "effect_1_CI95_upper",
        "effect_0_CI95_lower",
        "effect_0_CI95_upper",
    ]
    ensure_required_columns(df_with_arms, required_cols)

    # Compute effects based on type
    if effect_type in ["RR", "log-RR"]:
        # Synchronize inference with RD: use pooled logit-difference p-values
        # and derive RR for the effect axis.
        print(
            "\nComputing per-run RD pipeline outputs (for metadata and per-run RR)..."
        )
        df_per_run = compute_rd_pvalues(
            df_with_arms,
            group_cols=None,
            arm_pooling=args.arm_pooling,
            arm_pooling_rho=args.arm_pooling_rho,
            arm_weight_col=args.arm_weight_col,
            verbose=False,
        )
        # Derive per-run RR from arm probabilities
        if not df_per_run.empty:
            df_per_run = df_per_run.copy()
            df_per_run["RR"] = df_per_run["p1_hat"] / df_per_run["p0_hat"]

        print("Pooling arm logits and computing shared logit-difference p-values...")
        df_pooled = compute_rd_pvalues(
            df_with_arms,
            group_cols=("method", "outcome"),
            arm_pooling=args.arm_pooling,
            arm_pooling_rho=args.arm_pooling_rho,
            arm_weight_col=args.arm_weight_col,
            verbose=False,
        )
        # Derive pooled RR from pooled arm probabilities; keep p_value from logit t-test
        if not df_pooled.empty:
            df_pooled = df_pooled.copy()
            df_pooled["RR"] = df_pooled["p1_hat"] / df_pooled["p0_hat"]
        print(f"Computed {len(df_pooled)} method-outcome combinations")

        if df_pooled.empty:
            print("No pooled results available. Exiting.")
            return

        print("\nRisk ratio summary:")
        print(f"  RR range: [{df_pooled['RR'].min():.3f}, {df_pooled['RR'].max():.3f}]")
        print(f"  RR median: {df_pooled['RR'].median():.3f}")

    else:  # RD
        print("\nComputing per-run risk differences...")
        df_per_run = compute_rd_pvalues(
            df_with_arms,
            group_cols=None,
            arm_pooling=args.arm_pooling,
            arm_pooling_rho=args.arm_pooling_rho,
            arm_weight_col=args.arm_weight_col,
            verbose=False,
        )

        print("Pooling across runs using arm-level pooling on the logit scale...")
        df_pooled = compute_rd_pvalues(
            df_with_arms,
            group_cols=("method", "outcome"),
            arm_pooling=args.arm_pooling,
            arm_pooling_rho=args.arm_pooling_rho,
            arm_weight_col=args.arm_weight_col,
            verbose=False,
        )
        print(f"Computed {len(df_pooled)} method-outcome combinations")

        if "eta1_tau2" in df_pooled.columns or "RD" in df_pooled.columns:
            print("\nHeterogeneity statistics (tau²):")
            tau_cols = [c for c in ["eta1_tau2", "eta0_tau2"] if c in df_pooled.columns]
            if tau_cols:
                for col in tau_cols:
                    print(
                        f"  {col}: mean={df_pooled[col].mean():.4e}, "
                        f"max={df_pooled[col].max():.4e}"
                    )
            if "tau2" in df_pooled.columns:
                print(f"  Mean tau²: {df_pooled['tau2'].mean():.4e}")
                print(f"  Median tau²: {df_pooled['tau2'].median():.4e}")
                print(f"  Max tau²: {df_pooled['tau2'].max():.4e}")
                n_heterogeneous = (df_pooled["tau2"] > 0.01).sum()
                print(
                    f"  Cases with substantial heterogeneity (tau² > 0.01): "
                    f"{n_heterogeneous}"
                )

    # Run diagnostics if requested
    if args.diagnostics:
        run_diagnostics(
            df_pooled,
            df_with_arms,
            effect_type=effect_type,
            out_dir=str(args.output_dir),
        )

    # Prepare volcano plot data
    print("\nPreparing volcano plot data...")
    df_volcano = prepare_volcano_data(
        df_pooled,
        rd_col=effect_col,
        p_col="p_value",
        method_col="method",
        outcome_col="outcome",
        adjust=args.adjust,
        adjust_per=args.adjust_per,
        effect_alias=effect_alias,
    )

    # Load metadata
    atc_mapping = load_atc_dictionary()
    prevalence_stats: PrevalenceStats = load_prevalence_statistics(stats_path)
    prevalence_summary = rename_prevalence_columns(prevalence_stats.summary)
    per_run_summary = (
        summarise_per_run_effects(
            df_per_run, effect_col=effect_col, effect_alias=effect_alias
        )
        if not df_per_run.empty
        else pd.DataFrame()
    )

    df_volcano_enriched = augment_volcano_dataframe(
        df_volcano,
        df_pooled,
        prevalence_summary,
        per_run_summary,
        atc_mapping,
        effect_col=effect_col,
    )

    # Filter by minimum prevalence if specified
    if args.min_prevalence > 0.0:
        n_before = len(df_volcano_enriched)
        df_volcano_enriched = df_volcano_enriched[
            df_volcano_enriched["prevalence_mean_total"] >= args.min_prevalence
        ].copy()
        n_after = len(df_volcano_enriched)
        n_filtered = n_before - n_after
        print(
            f"\nFiltered {n_filtered} outcomes with prevalence < {args.min_prevalence:.4f}"
        )
        print(f"Remaining: {n_after} outcomes")

    # Summary statistics
    print("\nSummary statistics:")
    for method in df_volcano_enriched["method"].unique():
        d = df_volcano_enriched[df_volcano_enriched["method"] == method]
        n_sig = (d["q_value"] < DEFAULT_ALPHA).sum()
        print(
            f"  {method}: {len(d)} outcomes, {n_sig} significant (q < {DEFAULT_ALPHA})"
        )

    # Confusion matrix
    print("\nTMLE vs IPW significance confusion matrix:")
    confusion_result = print_significance_confusion_matrix(
        df_volcano_enriched,
        methods=("TMLE", "IPW"),
        method_col="method",
        outcome_col="outcome",
        q_col="q_value",
        alpha=DEFAULT_ALPHA,
        return_confusion=True,
    )

    ensure_output_directory(args.output_dir)

    # Save confusion matrix as heatmap
    if confusion_result:
        confusion_df, agreement, n_overlap = confusion_result

        fig_cm = plot_confusion_matrix(
            confusion_df,
            agreement,
            n_overlap,
            method_a="TMLE",
            method_b="IPW",
        )

        confusion_png = args.output_dir / f"confusion_matrix_{output_suffix}.png"
        fig_cm.savefig(confusion_png, dpi=300, bbox_inches="tight")
        print(f"Saved confusion matrix to: {confusion_png}")
        plt.close(fig_cm)

    outcome_label_map = (
        df_volcano_enriched.dropna(subset=["outcome_label"])
        .drop_duplicates("outcome")
        .set_index("outcome")["outcome_label"]
        .to_dict()
    )

    # Create overlay plot (Matplotlib)
    print("\nCreating TMLE vs IPW overlay plot...")
    try:
        fig_overlay, _ = volcano_overlay_methods(
            df_volcano_enriched,
            methods=("TMLE", "IPW"),
            method_col="method",
            outcome_col="outcome",
            label_map=outcome_label_map,
            annotate_top_n=10,
            point_size=45,
            effect_col=effect_col,
            effect_label=effect_label,
        )
        overlay_png = (
            args.output_dir / f"volcano_plot_tmle_ipw_overlay_{output_suffix}.png"
        )
        fig_overlay.savefig(overlay_png, dpi=300, bbox_inches="tight")
        print(f"Saved overlay plot to: {overlay_png}")
        plt.close(fig_overlay)
    except ValueError as err:
        print(f"Skipping TMLE vs IPW overlay plot: {err}")

    # Create overlay plot (Plotly)
    print("\nCreating TMLE vs IPW overlay plot (interactive)...")
    try:
        plotly_overlay = build_plotly_overlay_methods(
            df_volcano_enriched,
            methods=("TMLE", "IPW"),
            method_col="method",
            outcome_col="outcome",
            label_map=outcome_label_map,
            marker_size=9,
            effect_col=effect_col,
            effect_label=effect_label,
            null_value=null_value,
            xscale=xscale,
        )
        overlay_html = (
            args.output_dir
            / f"volcano_plot_tmle_ipw_overlay_{output_suffix}_interactive.html"
        )
        save_plotly_figure(
            plotly_overlay,
            html_path=overlay_html,
            png_path=None,
        )
        print(f"Saved interactive overlay to: {overlay_html}")
    except ValueError as err:
        print(f"Skipping Plotly TMLE vs IPW overlay: {err}")

    # Create IPW vs TMLE correlation plot for the chosen effect
    print("\nCreating IPW vs TMLE correlation plot...")
    try:
        # For log-RR, display correlation on log10 scale (linear axes), for clarity
        corr_transform = "log10" if xscale in {"log"} else None
        fig_corr, ax_corr = plot_method_correlation(
            df_volcano_enriched,
            methods=("IPW", "TMLE"),
            method_col="method",
            outcome_col="outcome",
            effect_col=effect_col,
            effect_label=effect_label,
            xscale="linear",
            transform=corr_transform,
            clip_quantiles=(0.005, 0.995),
            hexbin=False,
            point_size=22,
            alpha=0.65,
            significance_col="q_value",
            alpha_threshold=DEFAULT_ALPHA,
        )
        corr_png = args.output_dir / f"correlation_{output_suffix}.png"
        fig_corr.savefig(corr_png, dpi=300, bbox_inches="tight")
        print(f"Saved IPW vs TMLE correlation plot to: {corr_png}")
        plt.close(fig_corr)
    except ValueError as err:
        print(f"Skipping correlation plot: {err}")

    # Create main volcano plot (Matplotlib)
    print("\nCreating volcano plot...")
    fig, axes = volcano_plot_per_method(
        df_volcano_enriched,
        alpha=DEFAULT_ALPHA,
        method_col="method",
        outcome_col="outcome",
        max_labels_per_panel=10,
        figsize_per_panel=(7, 5),
        point_size=20,
        sig_color="#d62728",
        ns_color="#7f7f7f",
        effect_col=effect_col,
        effect_label=effect_label,
        null_value=null_value,
        xscale=xscale,
    )

    output_path_png = args.output_dir / f"volcano_plot_{output_suffix}.png"
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {output_path_png}")

    # Create interactive plot (Plotly)
    print("\nCreating interactive Plotly volcano plot...")
    plotly_fig = build_plotly_volcano(
        df_volcano_enriched,
        alpha=DEFAULT_ALPHA,
        method_col="method",
        outcome_col="outcome",
        effect_col=effect_col,
        effect_label=effect_label,
        null_value=null_value,
        xscale=xscale,
    )

    plotly_html = args.output_dir / f"volcano_plot_{output_suffix}_interactive.html"
    save_plotly_figure(plotly_fig, html_path=plotly_html, png_path=None)
    print(f"Saved interactive plot to: {plotly_html}")

    print("\nDone!")


if __name__ == "__main__":
    main()
