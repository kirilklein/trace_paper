"""Unified script to create volcano plots for treatment effect analysis.

Supports both Risk Difference (RD) and Risk Ratio (RR) analyses with configurable
parameters via command-line interface.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from trace.diagnostics import run_diagnostics
from trace.io import (
    PrevalenceStats,
    load_atc_dictionary,
    load_prevalence_statistics,
)
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
from trace.statistics import (
    combine_rr_random_effects_HKSJ,
    compute_rd_pvalues,
    compute_rr_from_arm_estimates,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
ALPHA = 0.05
METHODS_WITH_ARMS = ("IPW", "TMLE")


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def load_semaglutide_estimates(path: Path) -> pd.DataFrame:
    """Load the arm-level estimates CSV and drop unnamed columns."""
    df = pd.read_csv(path)
    unnamed_cols = [col for col in df.columns if col.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def filter_methods_with_arm_cis(
    df: pd.DataFrame, methods: Iterable[str]
) -> pd.DataFrame:
    """Restrict the dataframe to methods providing arm-level confidence intervals."""
    return df[df["method"].isin(list(methods))].copy()


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Print a diagnostic summary of missing values for required columns."""
    print("\nChecking for required columns...")
    for col in required:
        n_missing = df[col].isna().sum()
        print(f"  {col}: {n_missing} missing values")


def print_significance_confusion_matrix(
    df: pd.DataFrame,
    *,
    methods: tuple[str, str],
    method_col: str = "method",
    outcome_col: str = "outcome",
    q_col: str = "q_value",
    alpha: float = ALPHA,
    return_confusion: bool = False,
) -> tuple[pd.DataFrame, float, int] | None:
    """Print a confusion matrix comparing significance across two methods.

    When ``return_confusion`` is ``True``, the function additionally returns a
    tuple containing the formatted confusion matrix, the agreement proportion,
    and the number of overlapping outcomes used to build the matrix.
    """
    if len(methods) != 2:
        raise ValueError(
            "Exactly two methods are required to build the confusion matrix."
        )

    df_subset = (
        df[df[method_col].isin(methods)][[outcome_col, method_col, q_col]]
        .dropna(subset=[q_col])
        .copy()
    )

    if df_subset.empty:
        print("  No outcomes with valid q-values for the requested methods.")
        return None

    df_subset["is_significant"] = df_subset[q_col] < alpha

    pivot = df_subset.pivot_table(
        index=outcome_col,
        columns=method_col,
        values="is_significant",
        aggfunc="first",
    )

    missing = [method for method in methods if method not in pivot.columns]
    if missing:
        print(
            "  Missing methods in results: "
            + ", ".join(missing)
            + ". Not enough overlap to build the confusion matrix."
        )
        return None

    pivot = pivot.dropna(subset=list(methods))

    if pivot.empty:
        print("  No overlapping outcomes between the selected methods.")
        return None

    method_a, method_b = methods
    confusion = pd.crosstab(
        pivot[method_a],
        pivot[method_b],
        rownames=[f"{method_a} significant"],
        colnames=[f"{method_b} significant"],
        dropna=False,
    )

    confusion = confusion.reindex(
        index=[False, True], columns=[False, True], fill_value=0
    )

    formatted = confusion.rename(
        index={False: "No", True: "Yes"}, columns={False: "No", True: "Yes"}
    )
    print(formatted.to_string())

    agreement = (pivot[method_a] == pivot[method_b]).mean()
    print(f"  Agreement: {agreement * 100:.1f}% (n={len(pivot)})")

    if return_confusion:
        return formatted.copy(), agreement, len(pivot)

    return None


def summarise_per_run_effects(
    df_per_run: pd.DataFrame,
    *,
    effect_col: str = "RD",
    effect_alias: str | None = None,
) -> pd.DataFrame:
    """Aggregate per-run effect sizes and arm probabilities for hover metadata."""
    if df_per_run.empty:
        return pd.DataFrame()

    agg_kwargs = dict(
        per_run_n_runs=("run_id", "nunique"),
        per_run_effect1_mean=("effect_1", "mean"),
        per_run_effect1_std=("effect_1", "std"),
        per_run_effect0_mean=("effect_0", "mean"),
        per_run_effect0_std=("effect_0", "std"),
    )

    if effect_col in df_per_run.columns:
        prefix = (effect_alias or effect_col).lower()
        agg_kwargs.update(
            {
                f"per_run_{prefix}_mean": (effect_col, "mean"),
                f"per_run_{prefix}_median": (effect_col, "median"),
                f"per_run_{prefix}_std": (effect_col, "std"),
                f"per_run_{prefix}_min": (effect_col, "min"),
                f"per_run_{prefix}_max": (effect_col, "max"),
            }
        )

    grouped = (
        df_per_run.groupby(["method", "outcome"], dropna=False)
        .agg(**agg_kwargs)
        .reset_index()
    )
    return grouped


def rename_prevalence_columns(summary: pd.DataFrame) -> pd.DataFrame:
    """Make prevalence summary column names easier to work with."""
    rename_map = {
        "prevalence_overall_total": "prevalence_total",
        "prevalence_overall_treated": "prevalence_treated",
        "prevalence_overall_untreated": "prevalence_untreated",
        "population_total": "population_total",
        "population_treated": "population_treated",
        "population_untreated": "population_untreated",
        "outcome_events_total": "outcome_events_total",
        "outcome_events_treated": "outcome_events_treated",
        "outcome_events_untreated": "outcome_events_untreated",
        "prevalence_mean_total": "prevalence_mean_total",
        "prevalence_mean_treated": "prevalence_mean_treated",
        "prevalence_mean_untreated": "prevalence_mean_untreated",
        "prevalence_std_total": "prevalence_std_total",
        "prevalence_std_treated": "prevalence_std_treated",
        "prevalence_std_untreated": "prevalence_std_untreated",
        "prevalence_median_total": "prevalence_median_total",
        "prevalence_median_treated": "prevalence_median_treated",
        "prevalence_median_untreated": "prevalence_median_untreated",
        "run_count_total": "prevalence_run_count_total",
        "run_count_treated": "prevalence_run_count_treated",
        "run_count_untreated": "prevalence_run_count_untreated",
    }
    return summary.rename(
        columns={k: v for k, v in rename_map.items() if k in summary.columns}
    )


def augment_volcano_dataframe(
    volcano_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    prevalence_summary: pd.DataFrame,
    per_run_summary: pd.DataFrame,
    atc_mapping: Dict[str, str],
    *,
    effect_col: str = "RD",
) -> pd.DataFrame:
    """Merge pooled results with metadata for plotting and hover details."""
    # Avoid duplicating effect/p-values that already exist in volcano_df
    exclude_cols = {effect_col, "p_value"}
    pooled_meta = pooled_df.drop(
        columns=[c for c in exclude_cols if c in pooled_df.columns]
    )

    merged = volcano_df.merge(pooled_meta, on=["method", "outcome"], how="left")
    if not per_run_summary.empty:
        merged = merged.merge(per_run_summary, on=["method", "outcome"], how="left")
    if not prevalence_summary.empty:
        merged = merged.merge(prevalence_summary, on="outcome", how="left")

    merged["atc_description"] = merged["outcome"].map(atc_mapping)
    merged["outcome_label"] = merged["outcome"].where(
        merged["atc_description"].isna(),
        merged["outcome"] + " · " + merged["atc_description"],
    )
    return merged


def ensure_output_directory(path: Path) -> None:
    """Create output directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def print_dataset_overview(df: pd.DataFrame) -> None:
    """Print basic dataset statistics."""
    print(f"Loaded {len(df)} rows")
    print(f"Methods: {df['method'].unique()}")
    print(f"Outcomes: {df['outcome'].nunique()} unique")
    print(f"Runs: {df['run_id'].nunique()} unique")


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

    args = parser.parse_args()

    # Construct file paths
    estimates_path = args.input_dir / "combined_estimatest.txt"
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
    df_raw = load_semaglutide_estimates(estimates_path)
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
        print("\nComputing per-run risk ratios...")
        df_per_run = compute_rr_from_arm_estimates(
            df_with_arms, group_cols=None, verbose=False
        )

        print("Pooling risk ratios with DL + HKSJ adjustment...")
        df_pooled = combine_rr_random_effects_HKSJ(
            df_per_run, group_cols=("method", "outcome")
        )
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
            pooling_method="inverse_variance_arms",
            verbose=False,
        )

        print("Pooling risk differences using inverse variance on arms...")
        df_pooled = compute_rd_pvalues(
            df_with_arms,
            group_cols=("method", "outcome"),
            pooling_method="inverse_variance_arms",
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
        run_diagnostics(df_pooled, df_with_arms, effect_type=effect_type)

    # Prepare volcano plot data
    print("\nPreparing volcano plot data...")
    df_volcano = prepare_volcano_data(
        df_pooled,
        rd_col=effect_col,
        p_col="p_value",
        method_col="method",
        outcome_col="outcome",
        adjust="bh",
        adjust_per="by_method",
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

    # Summary statistics
    print("\nSummary statistics:")
    for method in df_volcano_enriched["method"].unique():
        d = df_volcano_enriched[df_volcano_enriched["method"] == method]
        n_sig = (d["q_value"] < ALPHA).sum()
        print(f"  {method}: {len(d)} outcomes, {n_sig} significant (q < {ALPHA})")

    # Confusion matrix
    print("\nTMLE vs IPW significance confusion matrix:")
    confusion_result = print_significance_confusion_matrix(
        df_volcano_enriched,
        methods=("TMLE", "IPW"),
        method_col="method",
        outcome_col="outcome",
        q_col="q_value",
        alpha=ALPHA,
        return_confusion=True,
    )

    ensure_output_directory(args.output_dir)

    # Save confusion matrix as heatmap
    if confusion_result:
        confusion_df, agreement, n_overlap = confusion_result

        # Create sklearn-style confusion matrix heatmap
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        im = ax_cm.imshow(confusion_df.values, cmap="Blues", aspect="auto")

        # Add colorbar
        plt.colorbar(im, ax=ax_cm)

        # Set ticks and labels
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["No", "Yes"])
        ax_cm.set_yticklabels(["No", "Yes"])

        # Add text annotations
        for i in range(2):
            for j in range(2):
                ax_cm.text(
                    j,
                    i,
                    int(confusion_df.values[i, j]),
                    ha="center",
                    va="center",
                    color="white"
                    if confusion_df.values[i, j] > confusion_df.values.max() / 2
                    else "black",
                    fontsize=16,
                    fontweight="bold",
                )

        # Labels and title
        ax_cm.set_xlabel("IPW Significant", fontsize=12, fontweight="bold")
        ax_cm.set_ylabel("TMLE Significant", fontsize=12, fontweight="bold")
        ax_cm.set_title(
            f"Significance Agreement\n{agreement * 100:.1f}% (n={n_overlap})",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )

        plt.tight_layout()

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

    # Create main volcano plot (Matplotlib)
    print("\nCreating volcano plot...")
    fig, axes = volcano_plot_per_method(
        df_volcano_enriched,
        alpha=ALPHA,
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

    output_path_pdf = args.output_dir / f"volcano_plot_{output_suffix}.pdf"
    fig.savefig(output_path_pdf, bbox_inches="tight")
    print(f"Saved plot to: {output_path_pdf}")

    # Create truncated plot for RD only (to exclude extreme p-values)
    if effect_type == "RD":
        print("\n" + "=" * 70)
        print("Creating truncated volcano plot (excluding clipped p-values)")
        print("=" * 70)

        df_volcano_truncated = df_volcano_enriched[
            df_volcano_enriched["neglog10p"] < 299
        ].copy()
        n_excluded = len(df_volcano_enriched) - len(df_volcano_truncated)
        pct_excluded = 100 * n_excluded / max(len(df_volcano_enriched), 1)
        print(
            f"\nExcluded {n_excluded} outcomes with clipped p-values "
            f"({pct_excluded:.1f}%)"
        )
        print(f"Remaining outcomes: {len(df_volcano_truncated)}")

        for method in df_volcano_truncated["method"].unique():
            d = df_volcano_truncated[df_volcano_truncated["method"] == method]
            n_sig = (d["q_value"] < ALPHA).sum()
            print(f"  {method}: {len(d)} outcomes, {n_sig} significant (q < {ALPHA})")

        print("\nCreating truncated volcano plot...")
        fig_trunc, axes_trunc = volcano_plot_per_method(
            df_volcano_truncated,
            alpha=ALPHA,
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

        output_path_trunc_png = (
            args.output_dir / f"volcano_plot_{output_suffix}_truncated.png"
        )
        fig_trunc.savefig(output_path_trunc_png, dpi=300, bbox_inches="tight")
        print(f"Saved truncated plot to: {output_path_trunc_png}")

    # Create interactive plot (Plotly)
    print("\nCreating interactive Plotly volcano plot...")
    plotly_fig = build_plotly_volcano(
        df_volcano_enriched,
        alpha=ALPHA,
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
